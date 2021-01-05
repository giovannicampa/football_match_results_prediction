import pandas as pd
import os
import numpy as np
import pickle


def prepare_data():
    """ Function that prepares the available match data

    The function loads all available csv files, processes them and stores them into a single file
    The dataframe is then saved locally, to prevent to do the preprocessing multiple times and is also returned by the function

    Meaning of abreviations used in the tables.

    FTHG: Full Time Home Team Goals
    FTAG: Full Time Away Team Goals
    
    FTR:  Full Time Result (H, A, D)
    
    HS:   Home Team Shots
    AS:   Away Team Shots
    
    HF:   Home Team Fowls
    AF:   Away Team Fowls

    HST:  Home Team Shots on Target
    AST:  Away Team Shots on Target

    HY:   Home Team Yellow Cards
    AY:   Away Team Yellow Cards
    """

    
    data_all_seasons = pd.DataFrame() # dataframe for all the data (features + target)


    # ---------------------------------------------------------------------------------------------------------------
    # - Calculating the features for each csv file in the folder
    # - At the end of the loop they will be put together

    # List of all csv files with the information of the matches
    csv_list = os.listdir('data_matches')


    for i, csv_file in enumerate(csv_list):
        
        season = pd.read_csv("data_matches/" + csv_file)
        season.Date.astype('category')
        
        match_days = pd.unique(season.Date)
        
        # Renaming the dates to match dates (instead of 01/08/17, we will have day 0,1,2...)
        for day in range(len(match_days)):
            season.Date[season.Date == match_days[day]] = day
        
        # Finding all team names and sorting alphabetically
        teams = pd.unique(season.HomeTeam)
        teams.sort()
        
        # rows: progressive match days, columns: team names, content: points
        points_table = np.zeros([len(match_days),len(teams)])
        
        # rows: progressive match days, columns: team names, content: ranking
        team_ranking_table = np.zeros([len(match_days),len(teams)])
        
        
        # - Generating the ranking and the score tables

        # [columns = teams, rows = match days (filled row wise)]
        for day in range(len(match_days)):
            
            # First table: nr of points by team every day
            for team in range(len(teams)):
                is_home = season.HomeTeam == teams[team]
                is_away = season.AwayTeam == teams[team]
                up_to_this_day = season.Date <= day
                
                points_as_home = np.sum(season.loc[(is_home & up_to_this_day), 'FTHG']) # Cumulative points of the team at home until today
                points_as_away = np.sum(season.loc[(is_away & up_to_this_day), 'FTAG']) # Cumulative points of the team while away until today
                
                points_table[day, team] = max(points_as_home, points_as_away)
            
            points_today = points_table[day, :]
            points_today = sorted(np.unique(points_today), reverse = True)
                
            
            # Second table: ranking of the team by points day by day (calculated using the first table)
            for point_class in range(len(points_today)):
                
                nr_teams_same_points = sum(points_table[day, :] == points_today[point_class])
                
                team_ranking_table[day, points_table[day,:] == points_today[point_class]] = point_class
                
        
        # Storing the two tables in dataframes for easier access        
        points_df = pd.DataFrame(points_table)
        points_df.columns = teams
        
        ranking_df = pd.DataFrame(team_ranking_table)
        ranking_df.columns = teams
        
        # The ranking that we calculated is the ranking at the end of the day, so we need to shift it of one day, to get
        # the teams' ranking on the day in which they played
        
        ranking_df.shift(periods = 1)
        

        # ---------------------------------------------------------------------------------------------------------------
        # - Initialising the table with the features that will be used for the prediciton model
        # Here the two previous tables will be united
        
        nr_matches = season.shape[0]
        
        # Generating feature names (same for home- and away-team)
        table_names = ['ranking',
                        'mean_points_per_match_tot',
                        'mean_points_per_match_in_location',
                        'matches_won_tot',
                        'mean_shots',
                        'mean_shot_on_target',
                        'mean_fouls',
                        'mean_yellow_cards',
                        'score']

        locations = ['_home','_away']
        
        table_names_all = []
        for name in table_names:
            for location in locations:
                table_names_all.append(name+location)
              
        
        season_table = np.zeros([nr_matches, len(table_names_all)]) # 1st 2 is the nr of teams, 2nd 2 is the target features (2 scores)
        
        season_data = pd.DataFrame(season_table)
        season_data.columns = table_names_all
        

        # ---------------------------------------------------------------------------------------------------------------        
        # - Calculating and filling up the feature tables
        
        for match in range(nr_matches):
            
            # This the progressive match number and is kept equal through all the dataframes
            match_day = season.loc[match, 'Date']
            
            team_home = season.loc[match, 'HomeTeam']
            team_away = season.loc[match, 'AwayTeam']
        
        
            # - Feature: team ranking
        
            season_data.ranking_home[match] = ranking_df.loc[match_day, team_home]
            season_data.ranking_away[match] = ranking_df.loc[match_day, team_away]
        
            
            # - Feature: mean scores in the same location
        
            # Logical indexes to find the various elements for which I want to calculate the mean
            idx_date = season.Date <= match_day                 # Happened before the current date
        
            idx_home_at_home = season.HomeTeam  == team_home    # happened for the home team at home
            idx_away_at_away = season.AwayTeam  == team_away    # happened for the away team away
            idx_home_at_away = season.AwayTeam  == team_home    # happened for the home team away
            idx_away_at_home = season.HomeTeam  == team_away    # happened for the away team at home
            
            # List of points when playing in the same location (home or away)
            points_home_at_home = season.loc[(idx_date & idx_home_at_home), 'FTHG'].values.tolist()
            points_away_at_away = season.loc[(idx_date & idx_away_at_away), 'FTAG'].values.tolist()
            
            # Mean points of each team playing in the same situation
            mean_pts_home_as_home = np.mean(points_home_at_home)
            mean_pts_away_as_away = np.mean(points_away_at_away)
        
            # Assigning to dataframe
            season_data.mean_points_per_match_in_location_home[match] = mean_pts_home_as_home
            season_data.mean_points_per_match_in_location_away[match] = mean_pts_away_as_away
        
        
            # - Feature: mean scores in all locations (total)
            
            # Points in the opposite situation
            points_home_at_away = season.loc[(idx_date & idx_home_at_away), 'FTAG'].values.tolist()
            points_away_at_home = season.loc[(idx_date & idx_away_at_home), 'FTHG'].values.tolist()
            

            # Mean points on all the matches for the home team (same and opposite situation)
            mean_pts_home_tot = np.nanmean(points_home_at_away + points_home_at_home)        
                

            # Mean points on all the matches for the away team (same and opposite situation
            mean_pts_away_tot = np.nanmean(points_away_at_home + points_away_at_away)
            season_data.mean_points_per_match_tot_home[match] = mean_pts_home_tot
            season_data.mean_points_per_match_tot_away[match] = mean_pts_away_tot
        

            # - Feature: matches_won
            season_data.matches_won_tot_home[match] = np.mean((season.loc[(idx_date & idx_home_at_home), 'FTR'].values == 'H').tolist() + (season.loc[(idx_date & idx_home_at_away), 'FTR'].values == 'A').tolist())
            season_data.matches_won_tot_away[match] = np.mean((season.loc[(idx_date & idx_away_at_away), 'FTR'].values == 'A').tolist() + (season.loc[(idx_date & idx_away_at_home), 'FTR'].values == 'H').tolist())


            # - Feature: mean shots (all locations)
            season_data.mean_shots_home[match] = np.mean(season.loc[(idx_date & idx_home_at_home), 'HS'].values.tolist() + season.loc[(idx_date & idx_home_at_away), 'AS'].values.tolist())
            season_data.mean_shots_away[match] = np.mean(season.loc[(idx_date & idx_away_at_away), 'AS'].values.tolist() + season.loc[(idx_date & idx_away_at_home), 'HS'].values.tolist())
            
            
            # - Feature: mean shot on target (all locations)    
            season_data.mean_shot_on_target_home[match] = np.mean(season.loc[(idx_date & idx_home_at_home), 'HST'].values.tolist() + season.loc[(idx_date & idx_home_at_away), 'AST'].values.tolist())
            season_data.mean_shot_on_target_away[match] = np.mean(season.loc[(idx_date & idx_away_at_away), 'AST'].values.tolist() + season.loc[(idx_date & idx_away_at_home), 'HST'].values.tolist())
            
            
            # - Feature: mean fouls (all locations)
            season_data.mean_fouls_home[match] = np.mean(season.loc[(idx_date & idx_home_at_home), 'HF'].values.tolist() + season.loc[(idx_date & idx_home_at_away), 'AF'].values.tolist())
            season_data.mean_fouls_away[match] = np.mean(season.loc[(idx_date & idx_away_at_away), 'AF'].values.tolist() + season.loc[(idx_date & idx_away_at_home), 'HF'].values.tolist())
            
            
#            # - Feature: mean corners (all locations)
#            
#            season_data.mean_corners_home[match] = np.mean(season.loc[(idx_date & idx_home_at_home), 'HC'].values.tolist() + season.loc[(idx_date & idx_home_at_away), 'AC'].values.tolist())
#            season_data.mean_corners_away[match] = np.mean(season.loc[(idx_date & idx_away_at_away), 'AC'].values.tolist() + season.loc[(idx_date & idx_away_at_home), 'HC'].values.tolist())
            
            
            # - Features: mean yellow cards (all locations)
            
            season_data.mean_yellow_cards_home[match] = np.mean(season.loc[(idx_date & idx_home_at_home), 'HY'].values.tolist() + season.loc[(idx_date & idx_home_at_away), 'AY'].values.tolist())
            season_data.mean_yellow_cards_away[match] = np.mean(season.loc[(idx_date & idx_away_at_away), 'AY'].values.tolist() + season.loc[(idx_date & idx_away_at_home), 'HY'].values.tolist())
            
            
            # - Assigning the target values
            season_data.score_home[match] = season.FTHG[match]
            season_data.score_away[match] = season.FTAG[match]
        
        
        # - Putting the dataframes for the prediction model together
        data_temp = season_data.iloc[5:]
        
        if i == 0:
            data_all_seasons = data_temp
        else:
            data_all_seasons = pd.concat([data_all_seasons, data_temp])
        
    
    # - Removing NaNs
    data_all_seasons.dropna(axis = 0, inplace = True)
    
    
    # Adding a column of the winners for each team
    winners = np.empty([data_all_seasons.shape[0],1], dtype = 'str')
    winners[data_all_seasons.score_home > data_all_seasons.score_away] = 'H'    # Home team wins
    winners[data_all_seasons.score_home < data_all_seasons.score_away] = 'A'    # Away team wins
    winners[data_all_seasons.score_home == data_all_seasons.score_away] = 'D'   # Draw
    
    data_all_seasons['winners'] = winners


    data_all_seasons.to_csv("data.csv", index = False)  # Saving data


    # - Data will also be returned as output of the function
    return season_data
    
    