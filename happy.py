import json
import pandas as pd
import csv


class DataMover():
    """
    PARSES TICKS BEFORE EVERY KILL AND SPLITS EVERY KILL TO CSV
    """

    def __init__(self):
        self.x = {}
        #sqlEngine = create_engine(dbconn)
        #self.dbConnection = sqlEngine.connect()
        self.n_rounds = None
        self.MatchId = None

    def read_json(self,file_path):
        with open(file_path) as f:
            x = json.load(f)
            self.x = x
            self.n_rounds = len(self.x["GameRounds"])
            self.MatchId = self.x["MatchId"]

    def get_pos_player(self, steamid, main_folder):
        """
        :param steamid:
        :return: None
        """
        # Write headers for file
        with open(f"{main_folder}{self.MatchId}.csv",'w',newline='\n')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(['sus','ActiveWeapon', 'AreaId', 'AreaName', 'Armor', 'DistToBombsiteA',
         'DistToBombsiteB', 'EquipmentValue', 'HasDefuse', 'HasHelmet', 'Hp',
         'Inventory', 'IsAirborne', 'IsAlive', 'IsDucking', 'IsFlashed',
         'IsScoped', 'IsWalking', 'Money', 'Name', 'SteamId', 'TICK',
         'TotalUtility', 'ViewX', 'ViewY', 'X', 'Y', 'Z'])


        for rounds in range(self.n_rounds):
            for frames in range(len(self.x["GameRounds"][rounds]["Frames"])):
                for team in range(2):
                    for player in range(4):
                        if team == 1:
                            team_index = "T"
                        else:
                            team_index = "CT"
                        try:
                            if str(self.x["GameRounds"][rounds]["Frames"][frames][team_index]["Players"][player]["SteamId"]) == steamid:

                                dikt = self.x["GameRounds"][rounds]["Frames"][frames][team_index]["Players"][player]
                                dikt["TICK"] = self.x["GameRounds"][rounds]["Frames"][frames]["Tick"]

                                df = pd.DataFrame.from_records(dikt)

                                df.to_csv(f"{main_folder}{self.MatchId}.csv", mode='a',header=False)
                        except Exception as e:
                            print(e)
                            """with open(f"{main_folder}{self.MatchId}.csv", 'a') as csvfile:
                                writer = csv.DictWriter(csvfile,fieldnames=dikt.keys())
                                writer.writeheader()
                                for data in dikt:
                                    writer.writerow(data)"""

    def get_kills_tick(self,kill_folder):
        """
        Gets all the kills in the match
        """
        # Headers
        with open(f"{kill_folder}{self.MatchId}.csv",'w',newline='\n')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(['sus','AssistedFlash', 'AssisterAreaId', 'AssisterAreaName', 'AssisterName',
       'AssisterSide', 'AssisterSteamId', 'AssisterTeam', 'AssisterX',
       'AssisterY', 'AssisterZ', 'AttackerAreaId', 'AttackerAreaName',
       'AttackerBlind', 'AttackerName', 'AttackerSide', 'AttackerSteamId',
       'AttackerTeam', 'AttackerViewX', 'AttackerViewY', 'AttackerX',
       'AttackerY', 'AttackerZ', 'Distance', 'IsFirstKill', 'IsFlashed',
       'IsHeadshot', 'IsTrade', 'IsWallbang', 'NoScope', 'PenetratedObjects',
       'PlayerTradedName', 'PlayerTradedSteamId', 'PlayerTradedTeam', 'Second',
       'ThruSmoke', 'TICK', 'VictimAreaId', 'VictimAreaName', 'VictimName',
       'VictimSide', 'VictimSteamId', 'VictimTeam', 'VictimViewX',
       'VictimViewY', 'VictimX', 'VictimY', 'VictimZ', 'Weapon'])


        for rounds in range(self.n_rounds):
            for k in range(len(self.x["GameRounds"][rounds]["Kills"])):
                df = pd.DataFrame.from_records(self.x["GameRounds"][rounds]["Kills"][k],index=[0])
                #df.to_sql(name='kill2', con=self.dbConnection, index=False, if_exists='append')
                df.to_csv(f"{kill_folder}{self.MatchId}.csv", mode='a', header=False)


    def split_df_by_kill(self,mainfile,killfile,steamid,out_folder,write_to_csv=False):
        """
        :param attacker_name:
        :return: List of dfs consisting of ticks up to the kill
        """
        retruning_list = []
        #df1 = pd.read_sql('csgo2',con=self.dbConnection)
        #df2 = pd.read_sql(f'SELECT * FROM kill2 WHERE AttackerSteamId="{steamid}"',con=self.dbConnection)
        df1 = pd.read_csv(mainfile)
        df2 = pd.read_csv(killfile)
        df2 = df2[df2["AttackerSteamId"] == int(steamid)]
        # drop dupes
        df2 = df2.drop_duplicates()
        for cnt, x in enumerate(range(len(df2))):
            # Get the next kill and append to list
            mintick = min(df2["TICK"])
            out_df = df1[df1['TICK'] <= mintick]
            #print(out_df)
            if len(out_df)>0:
                retruning_list.append(out_df)
                if write_to_csv == True:
                    out_df.to_csv(f"{out_folder}/{cnt}.csv")
            # Cut off kill from big dfs
            df1 = df1[df1['TICK'] > mintick]
            df2 = df2[df2['TICK'] > mintick]

        return retruning_list

def main(json_name,steamid):
    datamover = DataMover()
    datamover.read_json(f"C:/Users/emill/PycharmProjects/csgoparse/csgo/games/{json_name}")
    datamover.get_pos_player(f"{steamid}",f"D:/Users/emill/csgocheaters/games/")
    datamover.get_kills_tick(r'D:\Users\emill\csgocheaters\kills/')

    mainfile = r'D:\Users\emill\csgocheaters\games/testgame.csv'
    killfile = r'D:\Users\emill\csgocheaters\kills/testgame.csv'
    out_folder = r"D:\Users\emill\csgocheaters\singlekills/"
    datamover.split_df_by_kill(mainfile, killfile, steamid, out_folder, write_to_csv=True)


json_name = 'testgame.json'
steamid = "76561198977257843"
main(json_name,steamid)