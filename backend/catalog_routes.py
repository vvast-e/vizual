"""Admin auth, materials/colors CRUD, public catalog lists."""

from __future__ import annotations

import os
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from catalog_auth import create_access_token, get_current_admin, get_current_admin_optional, hash_password, verify_password
from db.deps import get_db
from db.models import Admin, Color, Material

router = APIRouter(tags=["catalog"])

ALLOWED_TEX_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(BACKEND_DIR, "..")),
)
TEXTURES_DIR = os.path.join(PROJECT_ROOT, "public", "textures")
COLORS_DIR = os.path.join(PROJECT_ROOT, "public", "colors")


def _scene_category_value(v: str) -> str:
    s = (v or "").strip().lower()
    if s not in ("interior", "exterior"):
        raise HTTPException(status_code=400, detail="scene_category must be interior or exterior")
    return s


def _safe_segment(s: str, default: str = "other") -> str:
    out = "".join(c for c in s if c.isalnum() or c in "-_").strip()
    return out or default


def material_row_to_api(m: Material) -> dict[str, Any]:
    price = float(m.price) if m.price is not None else None
    return {
        "id": str(m.id),
        "name": m.name,
        "filename": m.file_path,
        "category": m.material_category,
        "scene_category": m.scene_category,
        "url": f"/textures/{m.file_path}",
        "price": price,
        "size": m.size,
        "is_visible_in_visualizer": m.is_visible_in_visualizer,
    }


def color_row_to_api(c: Color) -> dict[str, Any]:
    price = float(c.price) if c.price is not None else None
    return {
        "id": str(c.id),
        "name": c.name,
        "scene_category": c.scene_category,
        "url": f"/colors/{c.file_path}",
        "price": price,
        "is_visible_in_visualizer": c.is_visible_in_visualizer,
    }


# ─── Auth ───────────────────────────────────────────────────────────


class AdminLoginBody(BaseModel):
    username: str
    password: str


class BootstrapBody(BaseModel):
    username: str
    password: str
    setup_secret: str


@router.post("/api/admin/auth/login")
def admin_login(body: AdminLoginBody, db: Session = Depends(get_db)) -> dict[str, str]:
    admin = db.scalar(select(Admin).where(Admin.username == body.username))
    if admin is None or not verify_password(body.password, admin.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not admin.is_active:
        raise HTTPException(status_code=403, detail="Inactive account")
    token = create_access_token(admin.id, admin.username)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/api/admin/bootstrap")
def bootstrap_first_admin(body: BootstrapBody, db: Session = Depends(get_db)) -> dict[str, Any]:
    expected = os.getenv("FIRST_ADMIN_SETUP_SECRET")
    if not expected:
        raise HTTPException(status_code=404, detail="Bootstrap disabled")
    if body.setup_secret != expected:
        raise HTTPException(status_code=403, detail="Invalid setup secret")
    existing = db.scalar(select(func.count()).select_from(Admin))
    if existing and int(existing) > 0:
        raise HTTPException(status_code=400, detail="Administrators already exist")
    admin = Admin(
        username=body.username.strip(),
        password_hash=hash_password(body.password),
        is_active=True,
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return {"id": admin.id, "username": admin.username}


# ─── Admins CRUD ────────────────────────────────────────────────────


class AdminCreateBody(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1)


class AdminPatchBody(BaseModel):
    is_active: bool | None = None


class AdminOut(BaseModel):
    id: int
    username: str
    is_active: bool

    class Config:
        from_attributes = True


@router.post("/api/admins", response_model=AdminOut)
def create_admin(
    body: AdminCreateBody,
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
) -> Admin:
    uname = body.username.strip()
    if db.scalar(select(Admin).where(Admin.username == uname)):
        raise HTTPException(status_code=409, detail="Username already exists")
    row = Admin(username=uname, password_hash=hash_password(body.password), is_active=True)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/api/admins", response_model=list[AdminOut])
def list_admins(
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
) -> list[Admin]:
    return list(db.scalars(select(Admin).order_by(Admin.id)))


@router.patch("/api/admins/{admin_id}", response_model=AdminOut)
def patch_admin(
    admin_id: int,
    body: AdminPatchBody,
    db: Session = Depends(get_db),
    current: Admin = Depends(get_current_admin),
) -> Admin:
    row = db.get(Admin, admin_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    if body.is_active is not None:
        if row.id == current.id and not body.is_active:
            raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
        row.is_active = body.is_active
    db.commit()
    db.refresh(row)
    return row


@router.delete("/api/admins/{admin_id}")
def delete_admin(
    admin_id: int,
    db: Session = Depends(get_db),
    current: Admin = Depends(get_current_admin),
) -> dict[str, Any]:
    if admin_id == current.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    row = db.get(Admin, admin_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(row)
    db.commit()
    return {"deleted": admin_id}


# ─── Materials ──────────────────────────────────────────────────────


def _materials_visibility_filter(
    stmt: Any,
    admin: Admin | None,
    visible_only: bool,
) -> Any:
    if admin is None:
        return stmt.where(Material.is_visible_in_visualizer.is_(True))
    if visible_only:
        return stmt.where(Material.is_visible_in_visualizer.is_(True))
    return stmt


def _apply_material_filters(
    stmt: Any,
    admin: Admin | None,
    visible_only: bool,
    scene_category: str | None,
    material_category: str | None,
    q: str | None,
) -> Any:
    stmt = _materials_visibility_filter(stmt, admin, visible_only)
    if scene_category:
        stmt = stmt.where(Material.scene_category == _scene_category_value(scene_category))
    if material_category:
        stmt = stmt.where(Material.material_category == material_category.strip())
    if q and q.strip():
        like = f"%{q.strip().lower()}%"
        stmt = stmt.where(func.lower(Material.name).like(like))
    return stmt


@router.get("/api/materials")
def list_materials(
    db: Session = Depends(get_db),
    admin: Admin | None = Depends(get_current_admin_optional),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    scene_category: str | None = None,
    material_category: str | None = None,
    q: str | None = None,
    visible_only: bool = Query(True, description="If true, only items visible in visualizer"),
) -> dict[str, Any]:
    id_subq = _apply_material_filters(
        select(Material.id), admin, visible_only, scene_category, material_category, q
    ).subquery()
    total = db.scalar(select(func.count()).select_from(id_subq)) or 0
    stmt = _apply_material_filters(
        select(Material), admin, visible_only, scene_category, material_category, q
    )
    stmt = stmt.order_by(Material.id.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = list(db.scalars(stmt))
    return {"materials": [material_row_to_api(m) for m in rows], "total": int(total)}


class MaterialPatchBody(BaseModel):
    name: str | None = None
    scene_category: str | None = None
    material_category: str | None = None
    price: float | None = None
    size: str | None = None
    is_visible_in_visualizer: bool | None = None


@router.post("/api/materials")
async def create_material(
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
    file: UploadFile = File(...),
    name: str = Form(""),
    scene_category: str = Form(...),
    material_category: str = Form(...),
    price: str | None = Form(None),
    size: str | None = Form(None),
    is_visible_in_visualizer: str = Form("true"),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")

    ext = Path(file.filename or "texture.png").suffix.lower()
    if ext not in ALLOWED_TEX_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    sc = _scene_category_value(scene_category)
    mc = _safe_segment(material_category, "other")
    os.makedirs(os.path.join(TEXTURES_DIR, mc), exist_ok=True)

    base_name = _safe_segment(
        "".join(c for c in (name or file.filename or "texture") if c.isalnum() or c in "-_ ").strip().replace(" ", "-").lower(),
        "texture",
    )
    dest_name = f"{base_name}-{uuid.uuid4().hex[:10]}{ext}"
    rel_path = f"{mc}/{dest_name}"
    abs_path = os.path.join(TEXTURES_DIR, rel_path)

    contents = await file.read()
    with open(abs_path, "wb") as f:
        f.write(contents)

    price_dec: Decimal | None = None
    if price is not None and str(price).strip() != "":
        try:
            price_dec = Decimal(str(price))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid price")

    vis = str(is_visible_in_visualizer).lower() in ("1", "true", "yes", "on")
    row = Material(
        name=(name.strip() or base_name.replace("-", " ").title()),
        scene_category=sc,
        material_category=mc,
        price=price_dec,
        size=(size.strip() if size else None) or None,
        file_path=rel_path.replace("\\", "/"),
        is_visible_in_visualizer=vis,
    )
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        try:
            os.remove(abs_path)
        except OSError:
            pass
        raise
    db.refresh(row)
    return material_row_to_api(row)


@router.patch("/api/materials/{material_id}")
def patch_material(
    material_id: int,
    body: MaterialPatchBody,
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
) -> dict[str, Any]:
    row = db.get(Material, material_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    if body.name is not None:
        row.name = body.name.strip()
    if body.scene_category is not None:
        row.scene_category = _scene_category_value(body.scene_category)
    if body.material_category is not None:
        row.material_category = _safe_segment(body.material_category, "other")
    if body.price is not None:
        row.price = Decimal(str(body.price))
    if body.size is not None:
        row.size = body.size.strip() or None
    if body.is_visible_in_visualizer is not None:
        row.is_visible_in_visualizer = body.is_visible_in_visualizer
    db.commit()
    db.refresh(row)
    return material_row_to_api(row)


@router.delete("/api/materials/{material_id}")
def delete_material(
    material_id: int,
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
) -> dict[str, Any]:
    row = db.get(Material, material_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    rel = row.file_path.replace("..", "").replace("\\", "/").strip("/")
    abs_path = os.path.join(TEXTURES_DIR, rel)
    db.delete(row)
    db.commit()
    if os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass
    return {"deleted": material_id}


# ─── Colors ─────────────────────────────────────────────────────────


def _colors_visibility_filter(stmt: Any, admin: Admin | None, visible_only: bool) -> Any:
    if admin is None:
        return stmt.where(Color.is_visible_in_visualizer.is_(True))
    if visible_only:
        return stmt.where(Color.is_visible_in_visualizer.is_(True))
    return stmt


def _apply_color_filters(
    stmt: Any,
    admin: Admin | None,
    visible_only: bool,
    scene_category: str | None,
    q: str | None,
) -> Any:
    stmt = _colors_visibility_filter(stmt, admin, visible_only)
    if scene_category:
        stmt = stmt.where(Color.scene_category == _scene_category_value(scene_category))
    if q and q.strip():
        like = f"%{q.strip().lower()}%"
        stmt = stmt.where(func.lower(Color.name).like(like))
    return stmt


@router.get("/api/colors")
def list_colors(
    db: Session = Depends(get_db),
    admin: Admin | None = Depends(get_current_admin_optional),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    scene_category: str | None = None,
    q: str | None = None,
    visible_only: bool = Query(True),
) -> dict[str, Any]:
    id_subq = _apply_color_filters(
        select(Color.id), admin, visible_only, scene_category, q
    ).subquery()
    total = db.scalar(select(func.count()).select_from(id_subq)) or 0
    stmt = _apply_color_filters(select(Color), admin, visible_only, scene_category, q)
    stmt = stmt.order_by(Color.id.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = list(db.scalars(stmt))
    return {"colors": [color_row_to_api(c) for c in rows], "total": int(total)}


class ColorPatchBody(BaseModel):
    name: str | None = None
    scene_category: str | None = None
    price: float | None = None
    is_visible_in_visualizer: bool | None = None


@router.post("/api/colors")
async def create_color(
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
    file: UploadFile = File(...),
    name: str = Form(""),
    scene_category: str = Form(...),
    price: str | None = Form(None),
    is_visible_in_visualizer: str = Form("true"),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")

    ext = Path(file.filename or "color.png").suffix.lower()
    if ext not in ALLOWED_TEX_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    sc = _scene_category_value(scene_category)
    os.makedirs(COLORS_DIR, exist_ok=True)

    base_name = _safe_segment(
        "".join(c for c in (name or file.filename or "color") if c.isalnum() or c in "-_ ").strip().replace(" ", "-").lower(),
        "color",
    )
    dest_name = f"{sc}-{base_name}-{uuid.uuid4().hex[:10]}{ext}"
    abs_path = os.path.join(COLORS_DIR, dest_name)
    rel_path = dest_name

    contents = await file.read()
    with open(abs_path, "wb") as f:
        f.write(contents)

    price_dec: Decimal | None = None
    if price is not None and str(price).strip() != "":
        try:
            price_dec = Decimal(str(price))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid price")

    vis = str(is_visible_in_visualizer).lower() in ("1", "true", "yes", "on")
    row = Color(
        name=(name.strip() or base_name.replace("-", " ").title()),
        scene_category=sc,
        price=price_dec,
        file_path=rel_path.replace("\\", "/"),
        is_visible_in_visualizer=vis,
    )
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        try:
            os.remove(abs_path)
        except OSError:
            pass
        raise
    db.refresh(row)
    return color_row_to_api(row)


@router.patch("/api/colors/{color_id}")
def patch_color(
    color_id: int,
    body: ColorPatchBody,
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
) -> dict[str, Any]:
    row = db.get(Color, color_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    if body.name is not None:
        row.name = body.name.strip()
    if body.scene_category is not None:
        row.scene_category = _scene_category_value(body.scene_category)
    if body.price is not None:
        row.price = Decimal(str(body.price))
    if body.is_visible_in_visualizer is not None:
        row.is_visible_in_visualizer = body.is_visible_in_visualizer
    db.commit()
    db.refresh(row)
    return color_row_to_api(row)


@router.delete("/api/colors/{color_id}")
def delete_color(
    color_id: int,
    db: Session = Depends(get_db),
    _: Admin = Depends(get_current_admin),
) -> dict[str, Any]:
    row = db.get(Color, color_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    rel = row.file_path.replace("..", "").replace("\\", "/").strip("/")
    abs_path = os.path.join(COLORS_DIR, rel)
    db.delete(row)
    db.commit()
    if os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass
    return {"deleted": color_id}
