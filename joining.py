import pandas as pd



df3 = pd.read_csv("killtest.csv")

print(df3.columns)
#df3 = df3.set_index('TICK')
df3 = df3[df3["AttackerName"] == "YungLV"]

victims = [x for x in df3["VictimName"]]
for x in victims:
    df = pd.read_csv("test.csv")

    df1 = df[df["Name"] == "YungLV"]
    df2 = df[df["Name"] == x]
    print(x)
    df1 = df1.set_index("TICK")
    if x == "Dan":
        df2.to_csv(f"danny.csv")
    df2 = df2[["TICK","X","Y","Z"]]
    df2=df2.set_index("TICK")
    df.coords['lon'] = (df.coords['lon'] + 180) % 360 - 180
    df = df.sortby(df.lon)
    #print(df1["TICK"])
    #print(df2["TICK"])
    big = df1.merge(df2,on='TICK')

    big.to_csv(f"meme{x}.csv")






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