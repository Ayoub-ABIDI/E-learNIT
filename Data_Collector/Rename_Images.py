import os

# Path to your main folder
base_path = r"C:\Users\ayoub\OneDrive\Desktop\TSL\tunisian_sign_dataset\processed"

# Allowed image extensions
valid_ext = [".jpg", ".jpeg", ".png"]

for subfolder in os.listdir(base_path):
    subfolder_path = os.path.join(base_path, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    print(f"\n📁 Processing folder: {subfolder}")

    # Get all images
    images = [
        f for f in os.listdir(subfolder_path)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]
    images.sort()

    # STEP 1 — rename to temporary names to avoid conflicts
    temp_names = []
    for idx, filename in enumerate(images):
        ext = os.path.splitext(filename)[1].lower()
        temp_name = f"__temp_{idx}{ext}"
        os.rename(
            os.path.join(subfolder_path, filename),
            os.path.join(subfolder_path, temp_name)
        )
        temp_names.append(temp_name)

    # STEP 2 — rename temp names to final names
    for idx, temp_name in enumerate(temp_names, start=1):
        ext = os.path.splitext(temp_name)[1].lower()
        final_name = f"{subfolder}({idx}){ext}"

        os.rename(
            os.path.join(subfolder_path, temp_name),
            os.path.join(subfolder_path, final_name)
        )
        print(f"✔ {final_name}")

print("\n✨ Done! All images safely renamed.")
