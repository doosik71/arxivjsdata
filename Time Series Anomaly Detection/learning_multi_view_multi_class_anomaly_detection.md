# Learning Multi-view Multi-class Anomaly Detection

Qianzi Yu, Yang Cao, Yu Kang (2025)

## 🧩 Problem to Solve

본 논문은 산업용 이상치 탐지(Industrial Anomaly Detection, IAD) 분야에서 여러 클래스의 객체를 하나의 모델로 처리하는 Multi-Class Anomaly Detection(MCAD)과, 한 객체를 여러 각도에서 촬영한 이미지를 사용하는 Multi-View 시나리오를 동시에 해결하고자 한다.

기존의 MCAD 모델들은 다음과 같은 이유로 Multi-View 환경에서 성능이 저하되는 문제를 겪는다. 첫째, 시점 데이터의 불일치(Inconsistency) 문제로 인해 특정 시점에서는 이상치가 보이지만 다른 시점에서는 보이지 않는 경우가 발생한다. 둘째, 서로 다른 뷰 사이의 상관관계나 상호 보완적인 정보를 충분히 활용하지 못한다. 셋째, 객체의 가장자리(Edge)에 위치한 이상치는 뷰 간의 정렬 문제로 인해 탐지가 어렵다.

따라서 본 연구의 목표는 다중 클래스와 다중 뷰 환경을 모두 수용할 수 있는 통합 프레임워크인 MVMCAD(Multi-View Multi-Class Anomaly Detection)를 제안하여, 뷰 간의 관계를 효과적으로 모델링하고 이상치 탐지 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Encoder-Decoder 구조를 기반으로 하되, 다중 뷰 데이터의 특성을 반영하기 위한 세 가지 핵심 모듈을 도입하는 것이다.

1. **Semi-Frozen Encoder (SFE):** Frozen Backbone 앞에 학습 가능한 Pre-encoder Prior Enhancement 메커니즘을 추가하여, 기존의 시각적 표현 능력을 유지하면서도 서로 다른 뷰의 데이터 분포에 빠르게 적응하도록 설계하였다.
2. **Anomaly Amplification Module (AAM):** 전역적인 토큰 상호작용을 모델링하고, 정상 영역과 유사한 특성을 가진 토큰을 억제함으로써 상대적으로 드문 이상치 신호를 증폭시킨다.
3. **Cross-Feature Loss (CFL):** Encoder의 얕은 층 특징과 Decoder의 깊은 층 특징(및 그 반대)을 교차 정렬함으로써, 저수준의 텍스처 이상과 고수준의 구조적 이상을 모두 민감하게 포착하도록 유도한다.

## 📎 Related Works

### 산업용 이상치 탐지 (Industrial Anomaly Detection)

기존 연구들은 주로 클래스별로 개별 모델을 학습시켰으나, 최근에는 배포 효율성을 위해 통합 모델(Unified Model)을 학습시키는 추세이다. 대표적으로 UniAD는 Transformer 기반 Encoder와 Dual-branch Decoder를 통해 다중 클래스를 처리한다. 또한 RD4AD(Reverse Distillation), SimpleNet, DeSTSeg, DiAD, MambaAD, Dinomaly 등이 제안되었으며, 이들은 주로 단일 뷰 이미지 탐지에 집중되어 있다.

### 다중 뷰 학습 (Multi-view Learning)

MVCNN이나 MVSTER와 같이 여러 뷰의 정보를 융합하여 인식 및 복원 성능을 높이는 연구들이 존재한다. 특히 MVAD는 다중 뷰 이상치 탐지를 위한 선구적인 프레임워크로 Multi-View Attention Selector(MVAS)를 제안하였으나, 다중 클래스(Multi-class) 시나리오를 처리하는 능력은 부족하다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인

본 모델은 Encoder-Decoder 구조를 따른다. 입력 이미지는 SFE를 통해 특징이 추출되고, AAM을 거쳐 이상치 신호가 증폭된 중간 특징 $f_m$이 생성된다. 학습 단계에서는 Decoder가 이 특징을 재구성하며 CFL을 통해 Encoder 특징과 정렬된다. 추론 단계에서는 Encoder 특징과 Decoder 재구성 특징 사이의 픽셀 단위 재구성 오차를 계산하여 이상치 맵(Anomaly Map)과 점수를 생성한다.

### Semi-Frozen Encoder (SFE)

입력 이미지 $x_0 \in \mathbb{R}^{B \times C \times H \times W}$에 대해, Frozen Backbone 앞에 학습 가능한 Prior 메커니즘을 둔다.
먼저 입력 이미지 $X$에 대해 가중치 정규화를 수행한다.
$$\tilde{X}_{c,h,w} = \text{BN}(X) = \gamma_c \cdot \frac{X_{c,h,w} - \mu_c}{\sigma_c}$$
여기서 가중치 $\alpha_c$를 다음과 같이 정의하여 적용한다.
$$\alpha_c = \frac{|\gamma_c|}{\sum_{k=1}^{C} |\gamma_k|}, \quad X^{ch}_{c,h,w} = \sigma(\alpha_c \cdot \tilde{X}_{c,h,w}) \cdot X_{c,h,w}$$
이후 채널 차원의 평균 $\beta_{h,w}$를 계산하여 특징 일관성을 높인 뒤, 최종적으로 Frozen Encoder $\varepsilon$에 입력하여 초기 특징 $f_i$를 얻는다.
$$f_i = \varepsilon(\text{Patch}(X^{\text{prior}}_{c,h,w}))$$

### Anomaly Amplification Module (AAM)

AAM은 정상 패턴을 억제하고 이상 신호를 증폭시킨다. 우선 표준 Attention 메커니즘을 통해 Query($Q$), Key($K$), Value($V$)를 생성하고 전역 문맥 특징 $F$를 얻는다.
$$F = W_F(\text{Softmax}(\frac{QK^\top}{\sqrt{d_k}})V)$$
이후 $F$를 토큰 차원으로 정규화하고, 학습 가능한 온도 파라미터 $\gamma$를 이용해 토큰 유사도 점수 $\text{Sim}$과 소프트 어텐션 분포 $\Pi$를 계산한다.
$$\Pi = \text{Softmax}(\text{Sim})$$
정상 영역의 지배적인 패턴을 억제하기 위해 역가중치(Inverse Weighting) 기반의 Attention Factor $\text{Att}$를 계산하여 최종 출력 $f_m$을 생성한다.
$$\text{Att} = \frac{1}{1 + (\Pi^\top \cdot F^2)}, \quad f_m = W_{\text{out}}(-(F \cdot \Pi) \cdot \text{Att})$$

### Cross-Feature Loss (CFL)

다양한 세맨틱 레벨의 이상치를 포착하기 위해 Encoder의 얕은 특징 $f_{e1}$을 Decoder의 깊은 특징 $f_2$와 정렬하고, Encoder의 깊은 특징 $f_{e2}$를 Decoder의 얕은 특징 $f_1$과 정렬한다. 유사도는 코사인 유사도를 기반으로 정의한다.
$$\text{Score} = 1 - \cos(z_1, z_2)$$
전체 특징 중 상위 10%의 유사도 점수를 가진 집합 $I$를 선택하여 다음과 같이 손실 함수를 계산한다.
$$\mathcal{L}_{\text{cross}} = \frac{1}{2} \left( \frac{1}{|I|} \sum_{i \in I} \text{Score}(f_{e1}, f_2) + \frac{1}{|I|} \sum_{i \in I} \text{Score}(f_{e2}, f_1) \right)$$

## 📊 Results

### 실험 설정

- **데이터셋:** Real-IAD (30개 카테고리, 객체당 5개 뷰).
- **지표:** 이미지 레벨(AUROC, AP, $F_1$-max), 픽셀 레벨(AUROC, AP, $F_1$-max, AUPRO).
- **구현:** ViT-Base/14 (DINOv2-R 사전 학습)를 Frozen Backbone으로 사용.

### 정량적 결과

제안 방법은 기존 SOTA 모델인 Dinomaly를 상회하는 성능을 보였다.

- **Image-level:** AUROC 91.0% / AP 88.6% / $F_1$-max 82.1% (SOTA 대비 AUROC +1.7%p 향상).
- **Pixel-level:** AUROC 99.1% / AP 43.9% / $F_1$-max 48.2% / AUPRO 95.2% (SOTA 대비 AUPRO +1.3%p 향상).

### Ablation Study 및 분석

SFE, AAM, CFL 세 가지 모듈을 모두 사용했을 때 가장 높은 성능을 기록하였다. 다만, 픽셀 레벨의 AP와 $F_1$-max가 일부 구성에서 약간 낮게 나타나는 현상이 관찰되었는데, 이는 AAM의 과도한 증폭(Over-amplification) 효과로 인해 특정 국소 영역이 과하게 강조되었기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 다중 뷰와 다중 클래스라는 복잡한 제약 조건 하에서도 통합 모델을 통해 높은 성능을 달성하였다. 특히 SFE를 통해 뷰 간의 데이터 분포 차이를 극복하고, AAM을 통해 희소한 이상치 신호를 효과적으로 부각시킨 점이 강점이다.

비판적으로 해석하자면, Transformer 기반의 Encoder-Decoder 구조를 사용함에 따라 다른 경량 모델들에 비해 추론 속도가 상대적으로 느리다는 한계가 있다. 또한, 저자 스스로 언급했듯이 AAM의 증폭 강도를 조절하는 적응형 전략(Adaptive strategy)이 부재하여 픽셀 레벨의 정밀도에서 일부 손실이 발생했다는 점은 향후 해결해야 할 과제이다. 하지만 ViT-Small부터 Large까지 백본 크기를 확장했을 때 성능이 꾸준히 향상되는 Scalability를 보여주어, 모델의 범용적 잠재력을 입증하였다.

## 📌 TL;DR

이 논문은 다중 뷰-다중 클래스 산업용 이상치 탐지를 위한 통합 프레임워크 **MVMCAD**를 제안한다. **SFE**(뷰 적응), **AAM**(이상치 신호 증폭), **CFL**(다중 스케일 특징 정렬)을 통해 Real-IAD 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 실제 산업 현장에서 여러 각도의 카메라 데이터를 통합적으로 처리하는 효율적인 검사 시스템 구축에 기여할 가능성이 높다.
