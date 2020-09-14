import pandas as pd
import numpy as np
import os

TRAIN_PATH = 'deep_regression/data/train.csv'
TEST_PATH = 'deep_regression/data/test.csv'
train_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), TRAIN_PATH)
test_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), TEST_PATH)
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_train['SalePrice']=np.log1p(df_train['SalePrice'])

# combining test and train data
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values

final_data = pd.concat((df_train, df_test)).reset_index(drop=True)
final_data.drop(['SalePrice'], axis=1, inplace=True)

# missing value treatment
# filling NA into None

for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond",
           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "MSSubClass", "MasVnrType"):
    final_data[col] = final_data[col].fillna("None")

# The area of the lot out front is likely to be similar to the houses in the local neighbourhood
#use the median value of the houses in the neighbourhood to fill
final_data["LotFrontage"] = final_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


#missing values with 0
for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1",
           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
           "BsmtFullBath", "BsmtHalfBath"):
    final_data[col] = final_data[col].fillna(0)



#Mode
final_data['MSZoning'] = final_data['MSZoning'].fillna(final_data['MSZoning'].mode()[0])
final_data['Electrical'] = final_data['Electrical'].fillna(final_data['Electrical'].mode()[0])
final_data['KitchenQual'] = final_data['KitchenQual'].fillna(final_data['KitchenQual'].mode()[0])
final_data['Exterior1st'] = final_data['Exterior1st'].fillna(final_data['Exterior1st'].mode()[0])
final_data['Exterior2nd'] = final_data['Exterior2nd'].fillna(final_data['Exterior2nd'].mode()[0])
final_data['SaleType'] = final_data['SaleType'].fillna(final_data['SaleType'].mode()[0])
final_data["Functional"] = final_data["Functional"].fillna(final_data['Functional'].mode()[0])

# Feature Engineeering

# Age of building
final_data['AgeWhenSold']  = final_data['YrSold'] - final_data['YearBuilt']
final_data['YearsSinceRemod']  = final_data['YrSold'] - final_data['YearRemodAdd']

# drop columns

final_data = final_data.drop(["YearBuilt"],axis=1)
final_data = final_data.drop(["YearRemodAdd"],axis=1)
final_data = final_data.drop(["YrSold"],axis=1)
final_data = final_data.drop(["GarageYrBlt"],axis=1)
# dropping utilities
final_data = final_data.drop(['Utilities'], axis=1)

final_data['MSSubClass'] = final_data['MSSubClass'].apply(str)

# Overall condition
final_data['OverallCond'] = final_data['OverallCond'].astype(str)
final_data['MoSold'] = final_data['MoSold'].astype(str)
# label encoder


from sklearn.preprocessing import LabelEncoder

columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
           'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
           'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
           'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
           'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in columns:
    lbl = LabelEncoder()
    lbl.fit(list(final_data[c].values))
    final_data[c] = lbl.transform(list(final_data[c].values))

print('Shape final_data: ', final_data.shape)
# Feature Engineering

# Overall quality of the house
final_data["OverallGrade"] = final_data["OverallQual"] * final_data["OverallCond"]
# Overall quality of the exterior
final_data["ExterGrade"] = final_data["ExterQual"] * final_data["ExterCond"]
# Overall kitchen score
final_data["KitchenScore"] = final_data["KitchenAbvGr"] * final_data["KitchenQual"]
# Overall fireplace score
final_data["FireplaceScore"] = final_data["Fireplaces"] * final_data["FireplaceQu"]


# Total number of bathrooms
final_data["TotalBath"] = final_data["BsmtFullBath"] + (0.5 * final_data["BsmtHalfBath"]) + final_data["FullBath"] + (0.5 * final_data["HalfBath"])
# Total SF for house
final_data["TotalSF"] = final_data["GrLivArea"] + final_data["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
final_data["FloorsSF"] = final_data["1stFlrSF"] + final_data["2ndFlrSF"]
# Total SF for porch
final_data["PorchSF"] = final_data["OpenPorchSF"] + final_data["EnclosedPorch"] + final_data["3SsnPorch"] + final_data["ScreenPorch"]
# Has masonry veneer or not
final_data["HasMasVnr"] = final_data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, "Stone" : 1, "None" : 0})
# House completed before sale or not
final_data["CompletedBFSale"] = final_data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
# Month Sold


final_data['mnth_sin'] = np.sin((final_data.MoSold-1)*(2.*np.pi/12))
final_data['mnth_cos'] = np.cos((final_data.MoSold-1)*(2.*np.pi/12))
final_data["TotalBuiltSF"] = final_data["GrLivArea"] + final_data["GarageArea"]+ final_data["OpenPorchSF"]+ final_data["PoolArea"]
final_data["BuiltInRatio"] = final_data["TotalBuiltSF"] /final_data["LotArea"]
final_data["GarageGrade"] = final_data["GarageQual"] * final_data["GarageCond"]

final_data = final_data.drop(["Exterior2nd"   ],axis=1)
final_data = final_data.drop(["Fence"   ],axis=1)
final_data = final_data.drop(["Functional"       ],axis=1)
final_data = final_data.drop(["GarageCond"       ],axis=1)
final_data = final_data.drop(["GarageFinish"      ],axis=1)
final_data = final_data.drop(["GarageQual"    ],axis=1)
final_data = final_data.drop(["KitchenAbvGr"       ],axis=1)
final_data = final_data.drop(["KitchenQual"       ],axis=1)
final_data = final_data.drop(["MiscFeature"       ],axis=1)

final_data = final_data.drop(["MiscVal"       ],axis=1)
final_data = final_data.drop(["OverallCond"       ],axis=1)
final_data = final_data.drop(["OverallQual"       ],axis=1)
final_data = final_data.drop(["Foundation"   ],axis=1)
final_data = final_data.drop(["BsmtFullBath"   ],axis=1)
final_data = final_data.drop(["BsmtHalfBath"   ],axis=1)
final_data = final_data.drop(["FullBath"       ],axis=1)
final_data = final_data.drop(["HalfBath"       ],axis=1)
final_data = final_data.drop(["GrLivArea"      ],axis=1)
final_data = final_data.drop(["TotalBsmtSF"    ],axis=1)
final_data = final_data.drop(["1stFlrSF"       ],axis=1)
final_data = final_data.drop(["2ndFlrSF"       ],axis=1)
final_data = final_data.drop(["TotalSF"       ],axis=1)
final_data = final_data.drop(["FloorsSF"       ],axis=1)

final_data = final_data.drop(["OpenPorchSF"       ],axis=1)
final_data = final_data.drop(["EnclosedPorch"       ],axis=1)
final_data = final_data.drop(["3SsnPorch"       ],axis=1)
final_data = final_data.drop(["ScreenPorch"       ],axis=1)
final_data = final_data.drop(["PorchSF"       ],axis=1)
final_data = final_data.drop(["GarageCars"   ],axis=1)
final_data = final_data.drop(["TotalBath"   ],axis=1)
final_data = final_data.drop(["Alley"       ],axis=1)
final_data = final_data.drop(["BsmtFinSF1"       ],axis=1)
final_data = final_data.drop(["BsmtFinSF2"      ],axis=1)
final_data = final_data.drop(["BsmtUnfSF"    ],axis=1)
final_data = final_data.drop(["GarageArea"       ],axis=1)
final_data = final_data.drop(["LowQualFinSF"       ],axis=1)
final_data = final_data.drop(["MoSold"       ],axis=1)
final_data = final_data.drop(["PoolArea"       ],axis=1)

final_data = final_data.drop(["PoolQC"       ],axis=1)
final_data = final_data.drop(["BsmtQual"       ],axis=1)

final_data = final_data.drop(["PavedDrive"       ],axis=1)
final_data = final_data.drop(["Street"       ],axis=1)
# corr = df_train.corr()
# corr.sort_values(["SalePrice"], ascending = True, inplace = True)
# corr = corr.SalePrice

final_data = pd.get_dummies(final_data)
print(final_data.shape)
final_data = final_data.fillna(final_data.mean())
train = final_data[:ntrain]
test = final_data[ntrain:]
train['SalePrice'] = y_train
print(train.shape)
print(test.shape)
TRAIN_PATH = 'deep_regression/data/train_clean.csv'
TEST_PATH = 'deep_regression/data/test_clean.csv'
train_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), TRAIN_PATH)
test_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), TEST_PATH)
train.to_csv(train_path, index=0)
test.to_csv(test_path, index=0)


