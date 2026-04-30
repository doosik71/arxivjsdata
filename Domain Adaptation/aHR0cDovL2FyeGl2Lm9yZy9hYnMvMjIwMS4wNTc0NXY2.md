# Deep Optimal Transport for Domain Adaptation on SPD Manifolds

Ce Ju, Cuntai Guan (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 뇌-컴퓨터 인터페이스(Brain-Computer Interface, BCI)의 EEG(뇌전도) 신호 분석에서 발생하는 **도메인 시프트(Domain Shift)** 문제이다. EEG 신호는 일반적으로 공간 공분산 행렬(Spatial Covariance Matrices)로 표현되며, 이러한 행렬들은 수학적으로 **대칭 양의 정부호(Symmetric Positive Definite, SPD) 매니폴드** 위에 존재한다.

문제의 중요성은 동일한 피험자라 할지라도 측정 세션(Session)이 달라지면 신호의 분포가 변하게 되어, 한 세션에서 학습된 모델을 다른 세션에 적용했을 때 성능이 크게 저하된다는 점에 있다. 기존의 도메인 적응(Domain Adaptation, DA) 방법론들은 이러한 공분산 행렬을 단순히 유클리드 공간의 데이터로 취급하여 적용하는 경우가 많았으며, 이는 SPD 매니폴드가 가진 고유의 기하학적 구조를 무시함으로써 최적의 성능을 내지 못하는 결과를 초래했다.

따라서 본 논문의 목표는 SPD 매니폴드의 기하학적 구조를 보존하면서, 소스 도메인과 타겟 도메인 간의 **주변 분포(Marginal Distribution)**와 **조건부 분포(Conditional Distribution)**를 동시에 정렬하는 새로운 기하학적 딥러닝 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **최적 운송(Optimal Transport, OT) 이론**을 **Log-Euclidean Metric** 기반의 SPD 매니폴드 기하학에 통합하는 것이다.

핵심 기여 사항은 다음과 같다.
1. **Deep Optimal Transport (DOT) 프레임워크 제안**: 딥 인코더를 통해 SPD 매니폴드 상의 비선형 매핑을 학습하고, 이를 통해 소스-타겟 도메인을 공유된 SPD 부분 공간으로 투영하여 정렬한다.
2. **Joint Distribution Adaptation**: 단순한 주변 분포 정렬을 넘어, 클래스별 조건부 분포까지 동시에 정렬함으로써 더 정밀한 도메인 적응을 수행한다.
3. **Riemannian Cost Function의 도입**: 유클리드 거리가 아닌 Log-Euclidean 거리의 제곱을 OT 비용 함수로 사용하여 SPD 매니폴드의 내재적 구조를 반영하였다.
4. **비선형 매핑의 학습**: 기존의 선형 또는 아핀 변환 기반 OT-DA와 달리, SPDNet 아키텍처를 활용하여 복잡한 비선형 변환을 학습할 수 있게 하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 그 한계점을 언급한다.
- **기존 OT-DA**: 유클리드 공간에서의 분포 정렬에는 효과적이지만, SPD 행렬과 같은 매니폴드 데이터에 직접 적용할 경우 기하학적 왜곡이 발생한다.
- **JDOT (Joint Distribution Optimal Transport)**: 주변 분포와 조건부 분포를 동시에 고려하여 정렬하지만, 주로 유클리드 공간의 아핀 함수를 사용한다.
- **SPD 매니폴드 상의 OT-DA (Yair et al.)**: SPD 매니폴드에 OT를 적용하려는 시도가 있었으나, 비용 함수로 유클리드 거리를 사용하여 리만 기하학적 특성을 충분히 반영하지 못했으며, 특히 조건부 분포의 시프트(Label Shift) 문제를 간과하였다.

본 연구는 이러한 한계를 극복하기 위해 **Log-Euclidean Metric**을 채택하여 계산 효율성을 높이고, 딥러닝 기반의 비선형 인코더를 통해 더 유연한 도메인 정렬을 가능하게 함으로써 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
DOT 프레임워크는 SPD 매니폴드 위의 데이터를 입력받아 비선형 변환을 거쳐 공유 잠재 공간 $Z \subset S_{++}$로 투영하는 구조를 가진다. 전체 과정은 **SPDNet** 아키텍처를 기반으로 하며, 다음과 같은 레이어들로 구성된다.
- **BiMap Layer**: $S \rightarrow W S W^T$ 변환을 통해 SPD 행렬을 변형한다.
- **ReEig Layer**: 고윳값 분해 후 정류(Rectification)를 통해 비선형성을 도입한다.
- **LogEig Layer**: SPD 행렬을 정체성 행렬(Identity matrix)에서의 접공간(Tangent space)으로 매핑하여 유클리드 연산이 가능하게 한다.

### 훈련 목표 및 손실 함수
DOT의 핵심은 주변 분포 적응(MDA)과 조건부 분포 적응(CDA)을 동시에 수행하는 통합 손실 함수를 최소화하는 것이다.

**1. 주변 분포 적응 (Marginal Distribution Adaptation, MDA)**
소스 도메인 $B_S$와 타겟 도메인 $B_T$의 Log-Euclidean 프레셰 평균(Fréchet mean) 사이의 거리를 최소화한다.
$$L_{MDA} := \|\log(w(B_S)) - \log(w(B_T))\|_F$$
여기서 $w(B)$는 배치 $B$에 대한 Log-Euclidean 프레셰 평균이다.

**2. 조건부 분포 적응 (Conditional Distribution Adaptation, CDA)**
각 클래스 $\ell$에 대해 소스 데이터 $B_{S\ell}$와 타겟 데이터 $B_{T\ell}$의 평균 간의 거리를 최소화하여 클래스별 정렬을 수행한다.
$$L_{CDA} := \|\log(w(B_{S\ell})) - \log(w(B_{T\ell}))\|_F$$
타겟 도메인의 레이블이 없는 경우, 베이스 모델을 통해 생성된 **의사 레이블(Pseudo-labels)**을 사용하여 클래스를 구분한다.

**3. 전체 목적 함수**
최종 손실 함수는 교차 엔트로피 손실($L_{CE}$)과 위에서 정의한 분포 정렬 손실의 가중 합으로 정의된다.
$$L_{DOT} := \alpha_1 L_{CE} + \alpha_2 L_{MDA}^2 + \alpha_3 L_{CDA}^2$$

### 추론 절차
학습된 딥 인코더 $\phi$를 통해 소스와 타겟 데이터를 동일한 부분 공간으로 투영한 후, 이 공간에서 학습된 분류기를 통해 타겟 도메인의 데이터를 분류한다.

## 📊 Results

### 실험 설정
- **데이터셋**: KU (54명), BNCI2014001 (9명), BNCI2015001 (12명) 등 3가지 공개 EEG 모터 이미지 데이터셋을 사용하였다.
- **평가 시나리오**: 세션 1($S_1$)에서 학습하고 세션 2($S_2$)에서 테스트하는 단방향 전이($S_1 \rightarrow S_2$) 설정을 적용하였다. (반-지도 및 비-지도 학습 포함)
- **비교 대상**:
    - **분류기**: SVM, MDRM(Minimum Distance to Riemannian Mean), SPDNet.
    - **전이 방법**: Parallel Transport (RCT, ROT), Deep Parallel Transport (RieBN), Optimal Transport (EMD, SPDSW, logSW).

### 주요 결과
- **정량적 성능**: 실험 결과, DOT 프레임워크는 특히 딥러닝 기반 분류기(SPDNet)와 결합했을 때 기존의 OT 방법론이나 병렬 전송(Parallel Transport) 방법보다 우수한 성능을 보였다.
- **메트릭의 영향**: Affine-Invariant Riemannian Metric ($g^{AIRM}$) 보다 Log-Euclidean Metric ($g^{LE M}$)을 사용했을 때 성능 향상 폭이 더 컸으며, 이는 딥러닝 모델의 학습 안정성과 계산 효율성 측면에서 유리함을 시사한다.
- **분포 정렬의 효과**: MDA 단독 적용보다 CDA를 함께 적용했을 때(MDA/CDA) 정밀한 정렬이 가능함을 확인하였다. 다만, 의사 레이블의 품질에 따라 CDA의 성능 변동성이 존재하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 SPD 매니폴드라는 특수한 기하학적 구조를 무시하지 않고, 이를 Log-Euclidean Metric과 OT 이론으로 풀어냈다는 점에서 학술적 가치가 높다. 특히, 복잡한 이산 OT 문제를 매 단계 풀지 않고, 프레셰 평균 간의 거리를 최소화하는 방식으로 최적 운송의 원리를 딥러닝 손실 함수에 녹여내어 계산 복잡도를 획기적으로 줄이면서도 효과적인 정렬을 달성하였다.

### 한계 및 비판적 논의
1. **의사 레이블 의존성**: CDA 단계에서 타겟 도메인의 레이블이 없으므로 의사 레이블을 사용하는데, 초기 예측 정확도가 낮을 경우 잘못된 정렬이 일어나는 **에러 전파(Error Propagation)** 문제가 발생할 수 있다.
2. **성능 향상 폭의 제한**: 전이 학습 기법을 적용했음에도 불구하고 성능 향상 폭이 때로는 완만하게 나타난다. 이는 도메인 시프트 외에도 EEG 신호 자체의 노이즈나 피험자 개별 특성 등 다른 요인이 성능에 큰 영향을 미치기 때문으로 분석된다.
3. **도메인 시프트 측정 지표의 부재**: 현재는 분류 정확도로만 성능을 평가하고 있으나, 실제 도메인 시프트의 강도를 정량적으로 측정하고 이를 분류 성능과 연결 짓는 체계적인 분석 프레임워크가 부족하다.

## 📌 TL;DR

본 논문은 EEG 공분산 행렬이 갖는 SPD 매니폴드 구조를 보존하며 도메인 시프트를 해결하는 **Deep Optimal Transport (DOT)** 프레임워크를 제안한다. Log-Euclidean Metric 기반의 OT 비용 함수와 딥 인코더를 결합하여 주변 및 조건부 분포를 동시에 정렬함으로써, 세션 간 변동성이 큰 BCI 데이터에서 분류 성능을 향상시켰다. 이 연구는 매니폴드 데이터의 기하학적 특성을 반영한 딥러닝 기반 도메인 적응의 새로운 방향성을 제시하며, 향후 의료 영상 및 뇌과학 데이터 분석 분야에 광범위하게 적용될 가능성이 크다.