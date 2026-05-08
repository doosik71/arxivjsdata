# Federated Learning for Data and Model Heterogeneity in Medical Imaging

Hussain Ahmad Madni, Rao Muhammad Umer, and Gian Luca Foresti (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석 분야의 Federated Learning(FL)에서 발생하는 **데이터 이질성(Data Heterogeneity)**과 **모델 이질성(Model Heterogeneity)** 문제를 동시에 해결하는 것을 목표로 한다.

실제 의료 환경에서 각 병원(클라이언트)은 고유한 개인정보 보호 정책과 비즈니스 요구사항으로 인해 서로 다른 아키텍처의 커스텀 모델을 사용하며, 수집되는 데이터 또한 Non-IID(non-independent and identically distributed) 특성을 갖는다. 특히, 데이터 내의 레이블 다양성(Label Diversity)과 노이즈는 모델의 수렴을 방해하고 전역 모델의 성능을 저하시킨다. 기존의 FL 방법론들은 데이터 이질성이나 모델 이질성 중 하나만을 다루는 경향이 있으며, 두 가지 문제를 동시에 효과적으로 해결하는 체계적인 접근법이 부족한 상황이다. 따라서 본 연구는 모델과 데이터의 이질성을 동시에 활용하여 전역 모델의 효율성을 높이는 MDH-FL(Exploiting Model and Data Heterogeneity in FL) 방법론을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Knowledge Distillation(지식 증류)**과 **Symmetric Loss(대칭 손실 함수)**를 결합하여 모델과 데이터의 이질성에서 오는 부정적인 영향을 최소화하는 것이다.

1. **모델 이질성 해결**: 서버에 존재하는 공공 데이터셋(Public Dataset)을 활용하여 각 클라이언트 모델의 출력값(Logits) 분포를 정렬한다. 이를 통해 모델 아키텍처가 다르더라도 서로의 지식 분포를 학습함으로써 모델 간의 간극을 줄인다.
2. **데이터 및 레이블 이질성 해결**: 레이블 노이즈에 강건한 Symmetric Loss를 도입하여, 잘못된 레이블로 인한 오버피팅을 방지하고 학습의 안정성을 확보한다.
3. **통합 프레임워크**: 모델의 출력 정렬과 데이터의 노이즈 억제를 동시에 수행하는 파이프라인을 구축하여, 실제 병원 환경과 유사한 이질적 시나리오에서 높은 성능을 입증한다.

## 📎 Related Works

기존의 FL 연구들은 주로 FedAvg나 FedProx와 같이 모델 파라미터를 직접 집계하는 방식을 사용하였으나, 이는 모든 클라이언트가 동일한 모델 아키텍처와 IID 데이터를 가지고 있다는 가정을 전제로 한다.

모델 이질성을 해결하기 위해 FedMD나 FedDF와 같은 Knowledge Distillation 기반 방법들이 제안되었으나, 이들은 공유 모델이나 상호 합의(Mutual Consensus)에 의존하는 경향이 있어 처리 오버헤드가 발생하거나 학습 방향 설정에 어려움이 있다. 또한, 데이터 이질성과 레이블 노이즈를 해결하기 위한 다양한 강건한 손실 함수(Robust Loss Functions) 연구가 진행되었으나, 대부분은 중앙 집중식 데이터 환경이나 단일 모델 구조를 가정하고 있어 FL 환경의 분산된 구조와 모델 다양성을 충분히 반영하지 못했다.

본 논문은 이러한 한계를 극복하기 위해 공공 데이터를 통한 지식 분포 정렬과 대칭 손실 함수를 통한 레이블 노이즈 억제를 동시에 적용함으로써 기존 방법론과의 차별성을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조

MDH-FL의 전체 프로세스는 **로컬 학습(Local Training)**과 **협업 학습(Collaborative Learning)**의 두 단계로 나뉜다. 로컬 학습 단계에서는 각 클라이언트가 자신의 프라이빗 데이터로 모델을 최적화하며, 협업 학습 단계에서는 서버의 공공 데이터를 통해 다른 클라이언트와의 지식 분포를 정렬한다.

### 1. 모델 이질성 해결 (Model Heterogeneity)

서버는 공공 데이터셋 $d^0 = \{x^0_i\}_{i=1}^{N_0}$를 보유하며, 각 클라이언트 $c_p$는 이 데이터를 통해 자신의 지식 분포 $D^t_{c_p} = f(d^0, \Theta^t_{c_p})$를 생성한다. 두 클라이언트 $c_{p1}, c_{p2}$ 간의 지식 분포 차이는 Kullback-Leibler(KL) Divergence를 통해 측정된다.

$$KL(D^e_{c_{p1}} || D^e_{c_{p2}}) = \sum D^e_{c_{p1}} \log\left(\frac{D^e_{c_{p1}}}{D^e_{c_{p2}}}\right)$$

클라이언트 $c_p$가 다른 모든 클라이언트들과의 분포 차이를 합산한 손실 함수 $L^{p,e}_{c_{pl}}$은 다음과 같다.

$$L^{p,e}_{c_{pl}} = \sum_{p'=1, p' \neq p}^{P} KL(D^e_{c_{p'}} || D^e_{c_p})$$

최종적으로 모델 파라미터 $\Theta^e_{c_p}$는 다음과 같이 업데이트된다.

$$\Theta^e_{c_p} \leftarrow \Theta^{e-1}_{c_p} - \alpha \nabla_{\Theta} \left( \frac{1}{P-1} \cdot L^{p,e-1}_{c_{pl}} \right)$$

여기서 $\alpha$는 학습률을 의미하며, 이 과정을 통해 각 클라이언트는 자신의 모델 아키텍처를 유지하면서도 타 클라이언트의 지식 분포에 맞게 출력을 정렬한다.

### 2. 데이터 및 레이블 이질성 해결 (Data and Label Heterogeneity)

레이블 노이즈에 대응하기 위해 **Symmetric Cross Entropy (SCE)**를 사용한다. 일반적인 Cross Entropy(CE) 손실 함수 $L_c$는 다음과 같다.

$$L_c = -\sum_{i=1}^{N} g(x_i) \log(p(x_i))$$

여기서 $g(x_i)$는 실제 레이블 분포, $p(x_i)$는 예측 분포이다. CE는 레이블 노이즈가 있을 때 오버피팅될 위험이 크다. 이를 보완하기 위해 예측값 $p$를 기준으로 실제 분포 $g$를 학습하는 Reverse Cross Entropy(RCE) $L_{rc}$를 도입한다.

$$L_{rc} = -\sum_{i=1}^{N} p(x_i) \log(g(x_i))$$

최종적인 Symmetric Loss $L_s$는 두 손실 함수의 가중 합으로 정의된다.

$$L_s = \lambda L_c + L_{rc}$$

$\lambda$는 노이즈에 대한 오버피팅을 조절하는 하이퍼파라미터이다. 로컬 모델 $\Theta^e_{pl}$은 이 $L_s$를 통해 다음과 같이 업데이트된다.

$$\Theta^e_{pl} \leftarrow \Theta^{e-1}_{pl} - \alpha \nabla_{\Theta} L^{p,e-1}_s(f(x_p, \Theta^{e-1}_{pl}), \tilde{y}_p)$$

## 📊 Results

### 실험 설정

- **데이터셋**: 백혈병 암 진단을 위한 단일 세포 분류 작업 수행. 서버에는 INT_20 데이터셋(공공 데이터)을, 클라이언트에는 Matek_19 데이터셋(프라이빗 데이터)을 배치하였다.
- **이질성 설정**:
  - **데이터**: Dirichlet 분포($\gamma=0.5$)를 사용하여 Non-IID 데이터셋을 구축하였다.
  - **모델**: ShuffleNet, ResNet10, MobileNetv2, ResNet12의 4가지 서로 다른 아키텍처를 클라이언트에 할당하였다.
  - **레이블**: Symmetric flip과 Pair flip 방식을 통해 $\mu \in \{0.1, 0.2, 0.3\}$의 노이즈 비율을 적용하였다.
- **비교 대상**: SL-FedL, FedDF, Swarm-FHE, FedMD.

### 결과 분석

1. **동질적 환경(Homogeneous Data/Labels)**: 데이터와 레이블에 노이즈가 없는 상황에서도 제안 방법이 평균 정확도 81.69%를 기록하며 다른 방법론(74%~78%)보다 우수한 성능을 보였다.
2. **이질적 환경(Heterogeneous Data/Labels)**: 레이블 노이즈 비율 $\mu$가 증가할수록 모든 방법의 성능이 하락하였으나, 제안 방법은 가장 완만한 하락 곡선을 그리며 높은 성능을 유지하였다.
    - Symmetric flip ($\mu=0.1$)에서 제안 방법은 83.69%의 정확도를 기록하여, FedMD(79.78%)보다 월등히 높았다.
    - Pair flip ($\mu=0.3$)의 극한 상황에서도 제안 방법은 73.94%의 정확도를 보였으며, 이는 다른 방법론들이 60% 후반에서 70% 초반에 머무는 것과 대조적이다.

## 🧠 Insights & Discussion

### 강점

본 논문은 모델 아키텍처의 다양성과 데이터 레이블의 오염이라는 두 가지 실무적 난제를 동시에 해결하려는 시도가 매우 돋보인다. 특히, 단순한 파라미터 평균화가 아닌 Knowledge Distillation을 통한 출력 분포 정렬 방식을 택함으로써, 클라이언트가 자신의 최적화된 모델 구조를 유지하면서도 협업 학습의 이점을 누릴 수 있게 하였다. 또한 Symmetric Loss의 도입은 의료 데이터 특유의 레이블 불확실성 문제를 효과적으로 억제하였다.

### 한계 및 논의사항

가장 큰 가정은 **서버가 적절한 공공 데이터셋($d^0$)을 보유하고 있어야 한다**는 점이다. 실제 의료 환경에서 모든 태스크에 대해 적절한 공공 데이터셋을 구하는 것은 어려울 수 있으며, 공공 데이터의 양과 질이 KL Divergence 기반의 정렬 성능에 직접적인 영향을 미칠 것으로 보인다. 또한, 모델의 이질성이 극심할 경우(예: 매우 작은 모델과 매우 큰 모델의 조합), Logits의 스케일 차이로 인해 단순히 KL Divergence만으로는 정렬이 충분하지 않을 가능성이 있으며, 이에 대한 온도(Temperature) 하이퍼파라미터 조정 등의 추가 연구가 필요할 것으로 판단된다.

## 📌 TL;DR

본 논문은 의료 영상 분석을 위한 FL 환경에서 **모델 아키텍처의 불일치**와 **데이터 레이블의 노이즈** 문제를 동시에 해결하는 **MDH-FL** 프레임워크를 제안한다. **Kullback-Leibler Divergence**를 이용해 서로 다른 모델 간의 지식 분포를 정렬하고, **Symmetric Loss**를 통해 레이블 노이즈에 의한 오버피팅을 방지한다. 실험 결과, 다양한 레이블 노이즈 시나리오에서도 기존 FL 방법론 대비 높은 강건성과 정확도를 입증하였으며, 이는 실제 병원 간 협력 학습 시스템 구축에 중요한 기여를 할 수 있을 것으로 기대된다.
