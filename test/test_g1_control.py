import mujoco

# CHANGE THIS PATH TO YOUR G1 XML
XML_PATH = "scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("=" * 50)
print("JOINTS")
print("=" * 50)

for i in range(model.njnt):
    print(f"{i}: {model.joint(i).name}")

print("\n" + "=" * 50)
print("ACTUATORS")
print("=" * 50)

for i in range(model.nu):
    print(f"{i}: {model.actuator(i).name}")

print("\nDone.")