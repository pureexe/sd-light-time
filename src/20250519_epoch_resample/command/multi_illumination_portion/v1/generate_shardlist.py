def generate_shardlist(
    laion_count,
    multi_illum_count,
    laion_base="/pure/f1/datasets/laion-shading/v4_webdataset/train/train-{:04d}.tar",
    multi_illum_base="/pure/f1/datasets/multi_illumination/real/v4_webdataset/train/train-{:04d}.tar",
    output_file="/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250510_webdataset_support/command/multi_illumination_portion/v1/shardlist/multilum_real_5k_laion_150k.txt"
):
    with open(output_file, "w") as f:
        for i in range(laion_count):
            f.write(laion_base.format(i) + "\n")
        for i in range(multi_illum_count):
            f.write(multi_illum_base.format(i) + "\n")
    print(f"✅ Wrote {laion_count + multi_illum_count} entries to {output_file}")


# ✏️ Customize the values here
if __name__ == "__main__":
    generate_shardlist(
        laion_count=150,           # Number of shards from laion-shading (e.g., 25 = 0000 to 0024)
        multi_illum_count=5      # Number of shards from multi_illumination (e.g., 10 = 0000 to 0009)
    )
