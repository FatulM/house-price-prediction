import json

import numpy as np
import pandas as pd


# ================================================================ #

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


houses_raw = pd.read_csv(
    f'input/train.csv',
    na_values=['', 'NA'],
    keep_default_na=False,
    index_col=0,
).convert_dtypes()

features = houses_raw.drop(columns=['SalePrice'], axis=1)
target = houses_raw['SalePrice'].copy()

features_names = features.columns.to_numpy()
nom_features = np.array(['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig',
                         'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                         'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                         'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType',
                         'PavedDrive', 'MiscFeature', 'MoSold', 'SaleType', 'SaleCondition'])
ord_features = np.array(['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
                         'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                         'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',
                         'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                         'PoolQC', 'Fence'])
num_features = np.array(['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1',
                         'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF',
                         'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars',
                         'GarageYrBlt', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea',
                         'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF',
                         'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF',
                         'YearBuilt', 'YearRemodAdd', 'YrSold'])

type_features = {
    'nominal': nom_features,
    'ordinal': ord_features,
    'numerical': num_features,
}

with open('data/type_features.json', 'w') as f:
    json.dump(type_features, f, indent=2, cls=NpEncoder)

feature_types: pd.Series = pd.Series(
    index=pd.Index(
        features_names,
        name='feature'
    ),
    data=[
        'nominal' if feature in type_features['nominal'] else
        'ordinal' if feature in type_features['ordinal'] else
        'numerical' if feature in type_features['numerical'] else
        str(1 / 0)
        for feature in features_names
    ],
    name='type',
)
feature_types.to_csv(
    'data/feature_types.csv',
    float_format='%g',
    na_rep="",
)

mSSubClass_cats = np.array([20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190])
mSZoning_cats = np.array(['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'])
street_cats = np.array(['Grvl', 'Pave'])
alley_cats = np.array(['Grvl', 'Pave', 'NA'])
lotShape_lvls = np.array(['Reg', 'IR1', 'IR2', 'IR3'])
landContour_cats = np.array(['Lvl', 'Bnk', 'HLS', 'Low'])
utilities_lvls = np.array(['AllPub', 'NoSewr', 'NoSeWa', 'ELO'])
lotConfig_cats = np.array(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])
landSlope_lvls = np.array(['Gtl', 'Mod', 'Sev'])
neighborhood_cats = np.array(['Blmngtn', 'Blueste', 'BrDale', 'BrkSide',
                              'ClearCr', 'CollgCr', 'Crawfor',
                              'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV',
                              'Mitchel', 'NAmes', 'NoRidge',
                              'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
                              'SWISU', 'Sawyer', 'SawyerW',
                              'Somerst', 'StoneBr', 'Timber', 'Veenker'])
condition1_cats = np.array(['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
condition2_cats = np.array(['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
bldgType_cats = np.array(['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'])
houseStyle_cats = np.array(['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'])
overallQual_lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
overallCond_lvls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
roofStyle_cats = np.array(['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'])
roofMatl_cats = np.array(['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'])
exterior1st_cats = np.array(['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace',
                             'CBlock', 'CemntBd',
                             'HdBoard', 'ImStucc', 'MetalSd', 'Other',
                             'Plywood', 'PreCast',
                             'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'])
exterior2nd_cats = np.array(['AsbShng', 'AsphShn', 'BrkComm',
                             'BrkFace', 'CBlock', 'CemntBd',
                             'HdBoard', 'ImStucc', 'MetalSd',
                             'Other', 'Plywood', 'PreCast',
                             'Stone', 'Stucco', 'VinylSd',
                             'Wd Sdng', 'WdShing'])
masVnrType_cats = np.array(['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'])
exterQual_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po'])
exterCond_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po'])
foundation_cats = np.array(['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'])
bsmtQual_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
bsmtCond_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
bsmtExposure_lvls = np.array(['Gd', 'Av', 'Mn', 'No', 'NA'])
bsmtFinType1_lvls = np.array(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
bsmtFinType2_lvls = np.array(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'])
heating_cats = np.array(['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'])
heatingQC_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po'])
centralAir_cats = np.array(['N', 'Y'])
electrical_cats = np.array(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'])
kitchenQual_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po'])
functional_lvls = np.array(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'])
fireplaceQu_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
garageType_cats = np.array(['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'])
garageFinish_lvls = np.array(['Fin', 'RFn', 'Unf', 'NA'])
garageQual_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
garageCond_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
pavedDrive_cats = np.array(['Y', 'P', 'N'])
poolQC_lvls = np.array(['Ex', 'Gd', 'TA', 'Fa', 'NA'])
fence_lvls = np.array(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'])
miscFeature_cats = np.array(['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'])
moSold_cats = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
saleType_cats = np.array(['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'])
saleCondition_cats = np.array(['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'])

nominal_cats = list((mSSubClass_cats, mSZoning_cats, street_cats,
                     alley_cats, landContour_cats, lotConfig_cats,
                     neighborhood_cats, condition1_cats,
                     condition2_cats, bldgType_cats,
                     houseStyle_cats, roofStyle_cats,
                     roofMatl_cats, exterior1st_cats,
                     exterior2nd_cats, masVnrType_cats,
                     foundation_cats, heating_cats, centralAir_cats,
                     electrical_cats, garageType_cats, pavedDrive_cats,
                     miscFeature_cats,
                     moSold_cats,
                     saleType_cats, saleCondition_cats))
ordinal_lvls = list([lotShape_lvls, utilities_lvls, landSlope_lvls,
                     overallQual_lvls, overallCond_lvls,
                     exterQual_lvls, exterCond_lvls, bsmtQual_lvls,
                     bsmtCond_lvls, bsmtExposure_lvls, bsmtFinType1_lvls,
                     bsmtFinType2_lvls, heatingQC_lvls, kitchenQual_lvls, functional_lvls,
                     fireplaceQu_lvls, garageFinish_lvls, garageQual_lvls,
                     garageCond_lvls, poolQC_lvls, fence_lvls])
cats_dict = dict(zip(nom_features, nominal_cats))
lvls_dict = dict(zip(ord_features, ordinal_lvls))
cats_lvls_dict = dict(cats_dict, **lvls_dict)

with open('data/feature_nominal_categories.json', 'w') as f:
    json.dump(cats_dict, f, indent=2, cls=NpEncoder)

with open('data/feature_ordinal_levels.json', 'w') as f:
    json.dump(lvls_dict, f, indent=2, cls=NpEncoder)

with open('data/feature_nonnumerical_values.json', 'w') as f:
    json.dump(cats_lvls_dict, f, indent=2, cls=NpEncoder)

val_map = {
    'BldgType': {
        '2fmCon': '2FmCon',
        'Duplex': 'Duplx',
        'Twnhs': 'TwnhsI',
    },
    'Exterior2nd': {
        'Brk Cmn': 'BrkComm',
        'CmentBd': 'CemntBd',
        'Wd Shng': 'WdShing',
    },
    'MSZoning': {
        'C (all)': 'C',
    },
}

mapped_features = features.replace(to_replace=val_map)

features_fixed = mapped_features.copy()

features_fixed.loc[332, 'BsmtFinType2'] = 'Unf'
features_fixed.loc[332, 'BsmtExposure'] = 'No'

na_fix_map_Alley = alley_cats[-1]  # 'NA'
na_fix_map_BsmtQual = bsmtQual_lvls[-1]  # 'NA'
na_fix_map_BsmtCond = bsmtCond_lvls[-1]  # 'NA'
na_fix_map_BsmtExposure = bsmtExposure_lvls[-1]  # 'NA'
na_fix_map_BsmtFinType1 = bsmtFinType1_lvls[-1]  # 'NA'
na_fix_map_BsmtFinType2 = bsmtFinType2_lvls[-1]  # 'NA'
na_fix_map_FireplaceQu = fireplaceQu_lvls[-1]  # 'NA'
na_fix_map_GarageType = garageType_cats[-1]  # 'NA'
na_fix_map_GarageFinish = garageFinish_lvls[-1]  # 'NA'
na_fix_map_GarageQual = garageQual_lvls[-1]  # 'NA'
na_fix_map_GarageCond = garageCond_lvls[-1]  # 'NA'
na_fix_map_PoolQC = poolQC_lvls[-1]  # 'NA'
na_fix_map_Fence = fence_lvls[-1]  # 'NA'
na_fix_map_MiscFeature = miscFeature_cats[-1]  # 'NA'
na_fix_map = {
    'Alley': {np.nan: na_fix_map_Alley},
    'BsmtQual': {np.nan: na_fix_map_BsmtQual},
    'BsmtCond': {np.nan: na_fix_map_BsmtCond},
    'BsmtExposure': {np.nan: na_fix_map_BsmtExposure},
    'BsmtFinType1': {np.nan: na_fix_map_BsmtFinType1},
    'BsmtFinType2': {np.nan: na_fix_map_BsmtFinType2},
    'FireplaceQu': {np.nan: na_fix_map_FireplaceQu},
    'GarageType': {np.nan: na_fix_map_GarageType},
    'GarageFinish': {np.nan: na_fix_map_GarageFinish},
    'GarageQual': {np.nan: na_fix_map_GarageQual},
    'GarageCond': {np.nan: na_fix_map_GarageCond},
    'PoolQC': {np.nan: na_fix_map_PoolQC},
    'Fence': {np.nan: na_fix_map_Fence},
    'MiscFeature': {np.nan: na_fix_map_MiscFeature},
}
features_fixed.replace(na_fix_map, inplace=True)


# noinspection PyPep8Naming
def fix_MasVnrNAs(df):
    df.loc[
        df['MasVnrArea'] == 0,
        'MasVnrType'
    ] = masVnrType_cats[3]  # None


fix_MasVnrNAs(features_fixed)

ftr_AgeRemodYr = features_fixed['YrSold'] - features_fixed['YearRemodAdd']
index = ftr_AgeRemodYr[ftr_AgeRemodYr < 0].index
features_fixed.loc[index, 'YearRemodAdd'] = features_fixed.loc[index, 'YearBuilt']

train_data = pd.concat([features_fixed, target.to_frame()], axis='columns', copy=True)
train_data.to_csv(
    'data/train.csv',
    float_format='%g',
    na_rep="",
)

test_raw = pd.read_csv(
    f'input/test.csv',
    na_values=['', 'NA'],
    keep_default_na=False,
    index_col=0,
).convert_dtypes()

test_features = test_raw.copy()
mapped_test_features = test_features.replace(to_replace=val_map)
test_features_fixed = mapped_test_features.copy()
test_features_fixed.replace(na_fix_map, inplace=True)
fix_MasVnrNAs(test_features_fixed)

test_data = test_features_fixed.copy()
test_data.to_csv(
    'data/test.csv',
    float_format='%g',
    na_rep="",
)


def cat_diff(df):
    non_numerics = df[np.union1d(ord_features, nom_features)]
    diff_dict = dict([(ftr_name,
                       np.setdiff1d(
                           non_numerics[ftr_name].value_counts().index.values,
                           cats_lvls_dict[ftr_name]))
                      for ftr_name in non_numerics.columns.values.tolist()])
    non_empty_diff_dict = dict([(key, diff_dict[key])
                                for key in diff_dict.keys()
                                if diff_dict[key].size != 0])

    return non_empty_diff_dict


assert len(cat_diff(mapped_features)) == 0
assert len(cat_diff(mapped_test_features)) == 0

all_data = pd.concat([train_data.drop(columns=['SalePrice']), test_data])
feature_numerical_extents = {
    feature: [all_data[feature].min(), all_data[feature].max()]
    for feature in num_features
}
with open('data/feature_numerical_extents.json', 'w') as f:
    json.dump(feature_numerical_extents, f, indent=2, cls=NpEncoder)
