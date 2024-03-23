import pandas as pd
df = pd.read_csv('perch_full.csv')
perch_full = df.to_numpy()
#print(perch_full)

# 타깃 데이터는 그대로 가져오기
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

#perch_full, perch_weight 데이터를 훈련, 테스트 세트로 나눔
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42)

# 사이킷 런 변환기 사용
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
# 9개의 특성 보여주세요~
print(poly.get_feature_names_out())

#테스트 세트 변환
test_poly = poly.transform(test_input)


#다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
# 테스트 세트도 확인해보기
print(lr.score(test_poly, test_target))

# 더 고차원으로 만들기 위하여 데이터 변형
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape) # 무려 55개의 특성 생성

#성능 결과 확인
lr.fit(train_poly, train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_poly, test_target))  # 특성의 개수가 많아지므로 선형 모델이 완벽하게 하려고 함 -> 오류 발생

#규제 (머신이 과도하게 학습하지 못하게 훼방) a. 릿지, b. 라쏘
# 표준 점수 변환기
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀 (계수를 곱한 갑사을 기준으로 규제)
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled,test_target))

#릿지 알파값 찾기
import matplotlib.pyplot as plt
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       #릿지 모델 만듬
       ridge = Ridge(alpha=alpha)
       #릿지 모델 훈련
       ridge.fit(train_scaled, train_target)
       # 훈련, 테스트 점수 리스트에 저장
       train_score.append(ridge.score(train_scaled, train_target))
       test_score.append(ridge.score(test_scaled,test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
#plt.show()

# 그래프 상에서 가장 적합 값으로  최종 모델 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled,test_target))


#라쏘 회귀 (계수의 절댓값을 기준으로 규제 적용)
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       lasso = Lasso(alpha=alpha, max_iter=10000)
       lasso.fit(train_scaled, train_target)
       train_score.append(lasso.score(train_scaled,train_target))
       test_score.append((lasso.score(test_scaled,test_target)))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
#plt.show()

lasso=Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled,test_target))

print(np.sum(lasso.coef_ == 0))