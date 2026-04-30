# Evaluation of Inference Attack Models for Deep Learning on Medical Data

Maoqiang Wu, Xinyue Zhang, Jiahao Ding, Hien Nguyen, Rong Yu, Miao Pan, Stephen T. Wong (2020)

## 🧩 Problem to Solve

본 논문은 의료 분야에서 딥러닝 모델의 확산에 따라 발생할 수 있는 개인정보 보호 및 프라이버시 침해 문제, 특히 추론 공격(Inference Attack)에 의한 데이터 유출 위험을 해결하고자 한다. 

의료 데이터는 환자의 전자 건강 기록(EHR), 생체 이미지 등 매우 민감한 정보를 포함하고 있다. 최근 연구들에 따르면 딥러닝 모델을 쿼리할 수 있는 권한이 있는 악의적인 공격자가 모델의 출력값이나 중간 단계의 활성화 값을 이용하여 학습에 사용된 원본 이미지나 텍스트 기록을 재구성할 수 있음이 밝혀졌다. 특히 의료 분야에서는 이러한 프라이버시 위협이 의료 기관 간의 데이터 공유를 저해하고, 결과적으로 의료 AI 연구의 발전을 늦추는 주요 원인이 된다. 따라서 본 논문의 목표는 의료 데이터에 대한 추론 공격의 취약성을 평가하고, 이를 효과적으로 방어할 수 있는 메커니즘을 제안하여 의료 딥러닝 모델의 안전성을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 의료 데이터셋을 대상으로 두 가지 주요 추론 공격 모델의 위험성을 실증적으로 분석하고, 이에 대한 실용적인 방어 기법을 제시한 점에 있다.

1. **의료 데이터에 대한 추론 공격 평가**: Attribute Inference Attack과 Model Inversion Attack을 의료 기록 및 의료 이미지 데이터에 적용하여, 모델이 민감한 환자 정보와 고해상도 의료 이미지를 유출할 수 있음을 증명하였다. 특히 의료 데이터에 대해 Model Inversion Attack을 평가한 것은 본 연구가 처음이다.
2. **효율적인 방어 메커니즘 제안**: 기존의 복잡한 암호화 방식(TEE, 동형 암호화 등) 대신, 계산 부담이 적고 적용이 간편한 Label Perturbation과 Model Perturbation 기법을 제안하여 프라이버시 유출을 효과적으로 억제하였다.

## 📎 Related Works

논문에서는 딥러닝 모델을 대상으로 하는 다양한 프라이버시 공격을 소개한다. 

- **Membership Inference Attack**: 특정 샘플이 학습 데이터셋에 포함되었는지 여부를 추론하는 공격이다.
- **Model Encoding Attack**: 공격자가 학습 데이터에 직접 접근하여 민감한 정보를 모델 내에 인코딩한 후 나중에 이를 추출하는 방식이다.
- **Attribute Inference Attack**: 데이터셋의 일부 속성이 주어졌을 때, 모델을 이용해 누락된 민감한 속성을 추론하는 공격이다.
- **Model Inversion Attack**: 모델의 출력이나 중간 특징값을 이용해 입력 데이터(주로 이미지)를 복원하는 공격이다.

기존의 방어 방법으로는 신뢰 실행 환경(TEE, Trusted Execution Environment)이나 동형 암호화(Homomorphic Encryption) 등이 있으나, 이는 특수한 하드웨어 지원이 필요하거나 연산 비용이 매우 높다는 한계가 있다. 또한 차분 프라이버시(Differential Privacy, DP)는 주로 학습 단계의 노이즈 추가에 집중하므로, 추론 단계에서 발생하는 본 논문의 공격 시나리오를 완벽히 방어하기에는 부적절하다고 명시하고 있다.

## 🛠️ Methodology

### 1. Attribute Inference Attack (AIA)
AIA는 환자의 의료 기록 중 일부 속성이 공개되었을 때, 모델의 예측값(Confidence Score)을 이용하여 알려지지 않은 민감한 속성 $x_d$를 추론하는 공격이다.

공격자는 다음과 같은 조건하에 동작한다.
- 데이터 샘플 $(x, y)$에서 $x$는 입력 속성, $y$는 라벨이다.
- 전체 $d$개의 속성 중 $x_1, x_2, \dots, x_{d-1}$은 알려져 있고, $x_d$가 공격 대상인 민감 속성이다.
- 공격자는 모델 $f(x)$에 접근할 수 있으며, 사전 확률(Prior Probabilities) 정보를 가지고 있다.
- 공격자의 목표는 다음의 사후 확률(Posterior Probability)을 최대화하는 $x_d$의 값을 찾는 것이다.
$$\max P(x_d | x_1, x_2, \dots, x_{d-1}, f(x))$$

### 2. Model Inversion Attack (MIA)
MIA는 모델의 중간 출력값 $v_0 = f_\theta(x_0)$를 통해 원본 의료 이미지 $x_0$를 복원하는 공격이다. 본 논문은 두 병원이 모델의 일부를 나누어 갖는 협력적 추론(Collaborative Inference) 시나리오를 가정하며, 공격자는 모델의 구조나 파라미터 $\theta$를 모르는 블랙박스 환경에서 공격을 수행한다.

공격자는 원본 모델 $f_\theta$의 역함수를 근사하는 역네트워크(Inverse Network) $g_\omega \approx f_\theta^{-1}$를 학습시킨다. 학습 절차는 다음과 같다.
1. **관찰 단계**: 원본 데이터와 동일한 분포를 가진 샘플 집합 $X = \{x_1, \dots, x_n\}$을 모델 $f_\theta$에 입력하여 중간 출력값 $V = \{f_\theta(x_1), \dots, f_\theta(x_n)\}$을 획득한다.
2. **학습 단계**: $V$를 입력으로, $X$를 타겟으로 하여 $g_\omega$를 학습시킨다. 이때 손실 함수로는 픽셀 공간에서의 $l_2$ norm을 사용한다.
$$l(\omega; X) = \frac{1}{n} \sum_{i=1}^{n} \| g_\omega(f_\theta(x_i)) - x_i \|_2^2$$
3. **복원 단계**: 타겟 데이터의 중간 출력 $v_0$를 학습된 $g_\omega$에 입력하여 복원된 이미지 $x'_0 = g_\omega(v_0)$를 얻는다.

### 3. Defense Mechanisms
- **Label Perturbation (AIA 방어)**: 예측 라벨에 무작위 응답(Randomized Response) 기법을 적용한다. 플리핑 확률(Flipping probability) $p$를 설정하여, 확률 $p$로 예측 라벨 $y$를 다른 클래스로 변경한다. 이는 공격자가 정확한 예측값을 얻지 못하게 하여 속성 추론 정확도를 떨어뜨린다.
- **Model Perturbation (MIA 방어)**: 모델의 파라미터 $\theta$(가중치 및 편향)에 가우시안 노이즈를 추가한다.
$$\theta = \theta + N(0, \sigma^2 I)$$
이렇게 하면 모델의 중간 출력값이 변동되어 공격자가 정확한 역매핑 함수 $g_\omega$를 학습하는 것을 방해하며, 결과적으로 복원된 이미지의 품질을 저하시킨다.

## 📊 Results

### 1. Attribute Inference Attack 실험
- **데이터셋**: Cardiovascular disease (70,000건), Heart disease (303건).
- **설정**: MLP 분류기(은닉층 2개, 각 100개 뉴런) 사용.
- **결과**: 방어 기법이 없을 때($p=0$) 공격 정확도가 높았으나, 플리핑 확률 $p$가 증가함에 따라 추론 정확도가 유의미하게 감소하였다. 다만, $p$가 너무 높으면 모델 자체의 테스트 정확도 또한 약간 하락하는 트레이드-오프 관계가 관찰되었다.

### 2. Model Inversion Attack 실험
- **데이터셋**: MIAS(유방 촬영술), CBIS-DDSM(유방 촬영술).
- **설정**: 6개 컨볼루션 층과 2개 완전 연결 층으로 구성된 CNN 사용. 모델 분할 지점을 2, 4, 6번째 층으로 설정하여 비교하였다.
- **평가 지표**: MSE(Mean-Square Error), PSNR(Peak Signal-to-Noise Ratio), SSIM(Structural Similarity Index).
- **결과**:
    - **분할 지점의 영향**: 분할 지점이 앞쪽 층(Layer 2)일수록 복원된 이미지의 품질이 매우 높았으며(SSIM $\approx 0.999$), 깊은 층(Layer 6)으로 갈수록 이미지가 흐릿해졌으나 여전히 해부학적 특징을 식별할 수 있는 수준이었다.
    - **방어 효과**: Model Perturbation을 적용했을 때, 노이즈 스케일($\sigma$)이 0.02에서 0.05로 증가함에 따라 복원 이미지의 MSE는 증가하고 PSNR과 SSIM은 크게 감소하여 공격 효율이 급격히 떨어졌다. 예를 들어 MIAS 데이터셋(Layer 4)에서 SSIM은 0.994(방어 없음) $\rightarrow$ 0.714 ($\sigma=0.02$) $\rightarrow$ 0.170 ($\sigma=0.05$)로 감소하였다.

## 🧠 Insights & Discussion

본 논문은 의료 딥러닝 모델이 단순히 결과값만 내놓는 것이 아니라, 모델 내부의 정보나 출력 확률값을 통해 원본 데이터의 민감한 정보를 역추적할 수 있다는 점을 성공적으로 입증하였다. 

특히 Model Inversion Attack의 경우, 협력적 추론 환경에서 전달되는 중간 특징값이 생각보다 훨씬 많은 정보를 담고 있으며, 이를 통해 환자의 유방 촬영 이미지를 높은 유사도로 복원할 수 있다는 점은 의료 AI 배포 시 심각한 보안 위협이 될 수 있음을 시사한다.

제안된 방어 기법들은 매우 단순한 노이즈 추가 방식임에도 불구하고 강력한 방어 성능을 보였다. 이는 복잡한 암호화 체계 없이도 파라미터나 라벨에 적절한 수준의 섭동(Perturbation)을 주는 것만으로도 공격자의 역매핑 학습을 효과적으로 방해할 수 있음을 보여준다. 다만, 모델의 정확도(Utility)와 프라이버시 보호 수준(Privacy) 사이의 균형을 맞추기 위한 적절한 하이퍼파라미터($p, \sigma$) 설정이 필수적이라는 점이 논의된다.

## 📌 TL;DR

이 논문은 의료 딥러닝 모델이 **Attribute Inference Attack**과 **Model Inversion Attack**에 취약하여 환자의 민감한 속성이나 의료 이미지가 유출될 수 있음을 실험적으로 증명하였다. 이를 방어하기 위해 **라벨 섭동(Label Perturbation)**과 **모델 파라미터 섭동(Model Perturbation)**이라는 간편하고 효율적인 방어책을 제안하였으며, 이를 통해 모델 성능 저하를 최소화하면서 프라이버시 유출을 효과적으로 막을 수 있음을 보였다. 이 연구는 향후 의료 AI 시스템 구축 시 프라이버시 보존 설계(Privacy-preserving design)의 중요성을 일깨우는 중요한 근거가 된다.