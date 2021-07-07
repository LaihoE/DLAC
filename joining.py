import numpy as np


import pandas as pd
from math import sqrt,asin,atan2,pi


df3 = pd.read_csv("killtest.csv")

print(df3.columns)
#df3 = df3.set_index('TICK')
df3 = df3[df3["AttackerName"] == "Lehtu"]



victims = [x for x in df3["VictimName"]]
for x in victims:
    df = pd.read_csv("test.csv")

    df1 = df[df["Name"] == "Lehtu"]
    df2 = df[df["Name"] == x]
    print(x)
    df1 = df1.set_index("TICK")

    df2 = df2[["TICK","X","Y","Z"]]
    df2=df2.set_index("TICK")

    # Convert 0-360 range to -180 - 180 range for easier trig
    df1['ViewX'] = (df1['ViewX'] + 180) % 360 - 180
    df1['ViewY'] = (df1['ViewY'] + 180) % 360 - 180

    big = df1.merge(df2,on='TICK')

    x_off = []
    y_off = []


    x1 = big["X_x"]
    y1 = big["Y_x"]
    z1 = big["Z_x"]

    x2 = big["X_y"]
    y2 = big["Y_y"]
    z2 = big["Z_y"]

    xdif = np.array(x2 - x1)
    ydif = np.array(y2 - y1)
    zdif = np.array(z2 - z1)

    print(zdif)
    rightLeft = np.arctan2(ydif, xdif) * (180 / pi)
    hypot3D = np.sqrt(zdif**2 + xdif**2 + ydif**2)
    print(hypot3D)

    upDown = (-np.arcsin(zdif / hypot3D)) * (180 / pi)

    big["X_Off_By_Degrees"] = rightLeft - big["ViewX"]
    big["Y_Off_By_Degrees"] = upDown - big["ViewY"]

    big.to_csv(f"nptest.csv")

"""# drop dupes
df2 = df2.drop_duplicates()
# Check how many files in output dir so to know what name to give file
n_files = len([name for name in os.listdir(out_folder)])
print("n_files",n_files)

for cnt, x in enumerate(range(len(df2))):
    # Get the next kill and append to list
    mintick = min(df2["TICK"])
    out_df = df1[df1['TICK'] <= mintick]

    if len(out_df)>0:
        retruning_list.append(out_df)
        if write_to_csv is True:
            out_df.to_csv(f"{out_folder}/singlekills/{n_files + cnt}.csv")

    # Cut off kill from big dfs
    df1 = df1[df1['TICK'] > mintick]
    df2 = df2[df2['TICK'] > mintick]"""