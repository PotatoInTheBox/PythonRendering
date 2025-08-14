width  = 1536
height = 1024
with open("skybox.raw", "rb") as f:
    raw = f.read()

# Remove alpha (RGBA → RGB)
rgb = bytearray()
for i in range(0, len(raw), 4):
    rgb.extend(raw[i:i+3])

with open("skyboxTex.bytes", "wb") as f:
    f.write(width.to_bytes(2, "little"))
    f.write(height.to_bytes(2, "little"))
    f.write(rgb)