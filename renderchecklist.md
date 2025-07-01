# ✅ Custom Renderer Checklist

## ✅ Phase 1: 2D Renderer (Foundations)

> Goal: Render 2D triangles using basic rasterization

### Setup
- [x] Create a framebuffer or pixel buffer (e.g. 2D array of RGB values)
- [x] Setup a windowing/drawing system (e.g. Pygame, tkinter)

### Triangle Drawing
- [x] Represent a triangle with 3 (x, y) points
- [x] Use bounding box to scan triangle area
- [x] Use interpolation or barycentric coordinates to fill pixels
- [x] Apply constant color per triangle

### Math Primitives
- [ ] 2D vector add/sub/dot/scalar multiply
- [ ] Vector normalization and length
- [ ] Interpolation: `interpolate(a, b, t)`

---

## ✅ Phase 2: Basic 3D Renderer (Wireframe & Projection)

> Goal: Render 3D wireframe model projected into 2D

### Math Core
- [ ] 3D vectors: add, sub, dot, cross, normalize
- [ ] Homogeneous 4D vectors (x, y, z, w)
- [ ] 4x4 matrix implementation

### Transformations
- [ ] Matrix-vector multiplication (4x4 × 4)
- [ ] Translation, rotation, and scaling matrices
- [ ] Matrix composition (model × view × projection)

### Projection
- [ ] Perspective projection matrix
- [ ] Apply homogeneous divide (divide by `w`)
- [ ] Convert NDC → screen coordinates

### Wireframe
- [ ] Load `.obj` files (vertices + triangle faces)
- [ ] Apply transform & projection to all vertices
- [ ] Draw lines between projected vertices

---

## ✅ Phase 3: Filled 3D Triangles & Depth

> Goal: Render shaded 3D triangles with depth and occlusion

### Triangle Filling
- [ ] Project all vertices to screen space
- [ ] Use scanline fill or barycentric fill
- [ ] Implement depth buffer (`z-buffer`) and compare `z`

### Backface Culling
- [ ] Compute normal via cross product
- [ ] Cull triangles facing away from camera

### Flat Lighting
- [ ] Normalize light direction vector
- [ ] Dot product of normal and light vector
- [ ] Convert to grayscale color

---

## ✅ Phase 4: Advanced Features (Optional)

> Add more advanced rendering features after core works

- [ ] Camera matrix (position + orientation)
- [ ] Perspective-correct interpolation
- [ ] Texture mapping
- [ ] Clipping (screen edges or near plane)
- [ ] Animation (per-frame transform)
- [ ] Soft shadows or ambient occlusion
