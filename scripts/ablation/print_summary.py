import pandas as pd
df = pd.read_csv("ablation_results.csv")
tail_df = df[df["step"] >= 40]
print("=== ABLATION RESULTS: AVERAGE PERFORMANCE (LAST 10 EPOCHS) ===\n")
print("BATCH SIZE SCALING (LR=1e-4, Depth=12)")
bs_df = tail_df[(tail_df["lr"] == 0.0001) & (tail_df["depth"] == 12)]
print(bs_df.groupby("eff_bs")[["Validation/In_Degree_Pass", "Validation/Out_Degree_Pass"]].mean().to_string())
print("\nNETWORK DEPTH SCALING (BS=16, LR=1e-4)")
depth_df = tail_df[(tail_df["eff_bs"] == 16) & (tail_df["lr"] == 0.0001)]
print(depth_df.groupby("depth")[["Validation/In_Degree_Pass", "Validation/Out_Degree_Pass"]].mean().to_string())
print("\nLEARNING RATE SCALING (BS=16, Depth=12)")
lr_df = tail_df[(tail_df["eff_bs"] == 16) & (tail_df["depth"] == 12)]
print(lr_df.groupby("lr")[["Validation/In_Degree_Pass", "Validation/Out_Degree_Pass"]].mean().to_string())
