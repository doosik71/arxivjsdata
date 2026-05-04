# A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation

Adrian Celaya, Beatrice Riviere, David Fuentes (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Imaging Segmentation) 분야에서 딥러닝 모델의 성능은 주로 Dice coefficient와 Hausdorff Distance (HD) 기반 지표를 통해 측정된다. 하지만 현재 널리 사용되는 대부분의 손실 함수들은 학습 과정에서 Dice coefficient나 이와 유사한 영역 기반(Region-based) 지표만을 최적화하는 경향이 있다.

이러한 접근 방식은 Dice coefficient에서는 높은 점수를 얻더라도, Hausdorff Distance 기반 지표에서는 낮은 정확도를 보이는 문제를 야기한다. 특히 종양 분할(Tumor Segmentation)과 같은 작업에서는 HD 지표가 매우 중요한데, Dice score가 높음에도 불구하고 HD 에러가 크다면 모델이 작은 종양을 제대로 감지하지 못했음을 의미할 수 있기 때문이다. 따라서 본 논문의 목표는 Hausdorff Distance를 효과적으로 최소화하면서도 수치적 안정성과 계산 효율성을 갖춘 새로운 손실 함수인 Generalized Surface Loss (GSL)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Hausdorff Distance를 줄이기 위한 새로운 손실 함수인 Generalized Surface Loss (GSL)의 설계 및 제안이다. GSL의 중심적인 설계 아이디어는 다음과 같다.

1.  **수치적 유계성(Boundedness):** 손실 함수 값을 $[0, 1]$ 구간으로 제한하여, 영역 기반 손실 함수와 결합하여 사용할 때 특정 손실이 전체 학습을 지배(dominate)하지 않도록 설계하였다.
2.  **계산 효율성:** 기존의 Hausdorff Loss가 매 에포크마다 예측 결과에 대한 Distance Transform Map (DTM)을 재계산해야 했던 것과 달리, GSL은 미리 계산된 ground truth의 DTM만을 사용하여 계산 비용을 획기적으로 줄였다.
3.  **클래스 불균형 해결:** 데이터셋 전체의 클래스별 픽셀 수의 역수를 이용한 가중치 항을 도입하여, 의료 영상에서 흔히 발생하는 클래스 불균형 문제를 해결하였다.

## 📎 Related Works

논문에서는 기존의 손실 함수를 영역 기반(Region-based)과 경계 기반(Boundary-based)의 두 가지 카테고리로 나누어 설명한다.

**1. 영역 기반 손실 함수 (Region-Based Losses)**
- **Dice Loss (DL):** 두 이진 집합 간의 전역적 중첩도(overlap)를 측정한다. 하지만 경계선의 세밀한 거리(HD)를 고려하지 않아, 전역적 중첩도는 높지만 경계 오차가 큰 결과가 나올 수 있다.
- **Generalized Dice Loss (GDL):** 클래스별 면적의 역수를 가중치로 부여하여 불균형한 분할 문제에 대응한다. 그러나 여전히 영역 기반 방식이므로 HD 지표 개선에는 한계가 있으며, 매 배치마다 가중치가 변하여 최적화가 어려울 수 있다.

**2. 경계 기반 손실 함수 (Boundary-Based Losses)**
- **Hausdorff Loss (HL):** Distance Transform Maps (DTM)를 사용하여 HD를 추정한다. 하지만 매 단계마다 예측 마스크의 DTM을 생성해야 하므로 계산 비용이 매우 높고, 메모리 사용량이 많아 nnUNet과 같은 패치 기반 파이프라인에 적용하기 어렵다.
- **Boundary Loss (BL):** Ground truth의 DTM을 사용하여 경계까지의 거리를 측정한다. HL보다 계산 효율적이지만, 손실 함수 값이 $\left(-\infty, \infty\right)$로 유계되지 않아(unbounded), 이미지 크기나 복셀 간격에 따라 최적화가 불안정해지며 영역 기반 손실 함수의 기여도를 무력화시킬 위험이 있다.

## 🛠️ Methodology

### 전체 시스템 구조 및 GSL 정의
본 연구에서는 nnUNet 아키텍처를 기본 모델로 사용하며, 영역 기반 손실 함수($L_{\text{region}}$)와 제안하는 경계 기반 손실 함수($L_{\text{gsl}}$)를 가중 결합하여 학습을 진행한다. 전체 손실 함수 $L$은 다음과 같이 정의된다.

$$L = \alpha L_{\text{region}} + (1 - \alpha) L_{\text{gsl}}$$

여기서 $L_{\text{region}}$으로는 Dice-CE (Dice Loss + Cross Entropy)가 사용된다.

### Generalized Surface Loss (GSL)
GSL은 예측값 $P$가 Ground truth $T$에 최대한 가까워지도록 유도하며, 그 과정에서 DTM의 절댓값을 회복하는 것을 목표로 한다. 수식은 다음과 같다.

$$L_{\text{gsl}} = 1 - \frac{\sum_{k=1}^C w_k \sum_{i=1}^N (D_i^k) (1 - |T_i^k + P_i^k|)^2}{\sum_{k=1}^C w_k \sum_{i=1}^N (D_i^k)^2}$$

- $C$: 세그멘테이션 클래스 수
- $N$: 총 픽셀(또는 복셀) 수
- $D_i^k$: $k$번째 클래스의 ground truth에 대한 $i$번째 복셀의 DTM 값
- $T_i^k, P_i^k$: 각각 ground truth와 예측된 세그멘테이션 마스크의 값
- $w_k$: 클래스 불균형을 해결하기 위한 가중치

가중치 $w_k$는 전체 데이터셋에서 각 클래스에 속하는 복셀 수 $N_k$의 역수를 정규화하여 미리 계산한다.

$$w_k = \frac{1/N_k}{\sum_{j=1}^C 1/N_j}$$

### 학습 절차 및 $\alpha$ 스케줄링
학습 초기에는 영역 기반 손실에 집중하고, 학습이 진행됨에 따라 경계 기반 손실의 비중을 높이기 위해 $\alpha$ 값을 $1$에서 $0$으로 감소시킨다. 본 논문에서는 세 가지 스케줄링 방식을 제안한다.

1.  **Linear Schedule:** $\alpha$를 선형적으로 감소시킨다.
2.  **Step Schedule:** 정해진 스텝 길이($h$)마다 $\alpha$를 단계적으로 감소시킨다.
3.  **Cosine Schedule:** 코사인 함수를 이용하여 $\alpha$를 부드럽게 감소시킨다.

## 📊 Results

### 실험 설정
- **데이터셋:** LiTS (간 및 종양 분할, CT 영상), BraTS (뇌종양 분할, MRI 영상)
- **모델:** nnUNet
- **평가 지표:** Dice coefficient ($\uparrow$), 95th percentile Hausdorff distance (HD95, $\downarrow$), Average Surface Distance (ASD, $\downarrow$)
- **비교 대상:** DL, Dice-CE, GDL, HL, BL

### 주요 결과
- **정량적 결과:** Table 1에 따르면, GSL은 LiTS와 BraTS 데이터셋 모두에서 HD95와 ASD 지표를 유의미하게 낮추었다. 특히 Dice coefficient는 기존 영역 기반 손실 함수들과 대등한 수준을 유지하면서도, 경계 정확도를 크게 향상시켰다.
- **정성적 결과:** 시각화 결과(Figure 3, 4)에서 GSL로 학습된 모델은 특히 어려운 케이스(difficult cases)에서 Dice-CE나 BL보다 훨씬 정교하고 매끄러운 경계 예측 결과를 생성함을 확인하였다.
- **스케줄러 분석:** LiTS 데이터셋에서는 스텝 길이 5인 Step schedule이 가장 좋은 성능을 보였다. BraTS의 경우 작업(Whole Tumor, Tumor Core, Enhancing Tumor)마다 최적의 스케줄러가 다르게 나타났다.

## 🧠 Insights & Discussion

본 연구의 결과는 GSL이 의료 영상 분할에서 HD 및 ASD 정확도를 높이는 유망한 대안임을 보여준다. 저자들은 GSL이 기존 경계 기반 손실 함수보다 뛰어난 성능을 보이는 이유를 **정규화(Normalization)**에서 찾는다. GSL은 손실 값을 $[0, 1]$ 범위로 제한함으로써 영역 기반 손실 함수와 수치적 스케일을 맞추었으며, 이는 최적화 관점에서 안정성을 제공한다. 특히 의료 영상 데이터에 내재된 가우시안 노이즈(CT)나 리시안 노이즈(MRI)와 같은 노이즈 환경에서도 정규화된 손실 함수가 모델의 강건성(robustness)을 높인다는 기존 연구 결과와 일맥상통한다.

또한, $\alpha$ 스케줄러의 선택이 성능에 영향을 미친다는 점을 발견하였다. Step schedule이 효과적인 이유는 최적화 과정에서 한 번에 하나의 하위 문제(subproblem)에 집중할 수 있게 하여 Adam과 같은 최적화 도구가 각 단계를 더 효과적으로 최소화할 수 있기 때문으로 해석된다.

다만, 본 논문에서는 $L_{\text{region}}$으로 Dice-CE만을 사용하였는데, 실험 결과에서 Dice Loss 단독 사용이 더 좋은 경우가 있었으므로, 다른 영역 기반 손실 함수와의 조합 가능성이 열려 있다. 또한 가중치 $w_k$ 계산 시 지수 $p > 1$을 도입하거나, DTM 자체에 기반한 가중치를 사용하는 등의 추가적인 개선 여지가 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 전역적 중첩도(Dice)는 높지만 경계 오차(Hausdorff Distance)가 큰 문제를 해결하기 위해 **Generalized Surface Loss (GSL)**를 제안한다. GSL은 $[0, 1]$ 범위로 정규화된 경계 손실 함수로, 계산 효율성이 높고 클래스 불균형을 처리할 수 있는 가중치 구조를 갖는다. nnUNet을 이용한 LiTS 및 BraTS 데이터셋 실험 결과, Dice score를 유지하면서도 HD95와 ASD를 크게 낮추어 정밀한 종양 분할 가능성을 입증하였다. 이 연구는 특히 정밀한 경계 예측이 필수적인 의료 진단 및 치료 계획 수립 연구에 중요하게 활용될 수 있다.