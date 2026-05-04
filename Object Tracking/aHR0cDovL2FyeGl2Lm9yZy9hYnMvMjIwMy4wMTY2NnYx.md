# Correlation-Aware Deep Tracking

Fei Xie, Chunyu Wang, Guangting Wang, Yue Cao, Wankou Yang, Wenjun Zeng (2022)

## 🧩 Problem to Solve

시각적 객체 추적(Visual Object Tracking, VOT)의 핵심 과제는 **강건성(Robustness)**과 **변별력(Discrimination power)**이라는 두 가지 상충하는 목표를 동시에 달성하는 것이다. 강건성은 대상 객체의 외형 변화가 심하더라도 이를 인식해내는 능력이며, 변별력은 배경에 존재하는 유사한 방해물(Distractor)을 걸러내고 타겟만을 정확히 식별하는 능력이다.

기존의 Siamese 계열 네트워크들은 템플릿 이미지와 검색 이미지에서 각각 특징을 추출하는 Siamese-like 구조를 사용한다. 이러한 방식은 특징 추출 단계에서 타겟과 검색 이미지 간의 상호작용이 없기 때문에, 추출된 특징이 개별 인스턴스의 특성을 충분히 반영하지 못하는 'Target-unaware' 특성을 가진다. 이로 인해 이후의 상관관계(Correlation) 연산 단계에서 타겟과 유사한 방해물을 효과적으로 구분해내지 못하는 '타겟-방해물 딜레마(Target-distractor dilemma)'가 발생하며, 이는 추적 성능의 한계로 이어진다. 본 논문의 목표는 특징 추출 단계부터 타겟에 의존적인(Target-dependent) 특징을 생성하여 이 문제를 해결하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **특징 추출 네트워크 내부에 상관관계 연산을 깊게 통합**하는 것이다. 이를 위해 **Single Branch Transformer (SBT)**라는 새로운 프레임워크를 제안한다.

1.  **Target-Dependent Feature Extraction**: 기존의 독립적인 특징 추출 방식에서 벗어나, Self-Attention(SA)과 Cross-Attention(CA)을 결합한 EoC(Extract-or-Correlation) 블록을 통해 템플릿과 검색 이미지의 특징이 네트워크 층을 통과하며 지속적으로 상호작용하도록 설계하였다.
2.  **Removal of Separate Correlation Step**: 특징 추출 과정에서 이미 깊은 수준의 상관관계 분석이 이루어지므로, 기존 추적기들이 사용하던 별도의 상관관계 연산(예: Siamese cropping, DCF 등) 단계 없이 검색 이미지의 특징만으로 즉시 타겟의 위치와 크기를 예측할 수 있다.
3.  **Flexible Pre-training**: SBT는 구조적으로 단일 스트림(Single-stream) 처리가 가능하여, ImageNet과 같은 대규모 비쌍(unpaired) 이미지 데이터셋에서 사전 학습이 가능하며, 이는 추적 작업으로의 미세 조정(Fine-tuning) 시 빠른 수렴 속도를 제공한다.

## 📎 Related Works

**1. Siamese Networks**: 강력한 백본 네트워크를 통해 표현력을 높였으나, 얕은 상관관계 구조로 인해 방해물에 대한 변별력이 부족하다. 이를 해결하기 위해 attention 메커니즘이나 온라인 업데이트 모듈 등이 추가되었으나, 파이프라인의 복잡도가 증가하는 단점이 있다.

**2. Discriminative Correlation Filter (DCF)**: 온라인으로 타겟 모델을 학습하여 변별력을 높이지만, 수작업으로 설계된 최적화 과정에 민감하며 복잡한 시나리오에서 인스턴스 수준의 변별력이 부족한 경우가 많다.

**3. Transformer-based Trackers**: Transformer의 장거리 모델링 능력을 활용해 특징을 융합하지만, 주로 Transformer를 특징 추출 이후의 융합 모듈(Fusion module)로 사용한다. 또한, 비전 작업에 적합한 초기화가 어려워 학습 비용이 매우 크다는 한계가 있다.

SBT는 이러한 기존 방식들과 달리, 특징 추출 단계부터 $\text{Cross-Attention}$을 통해 타겟-의존적 특징을 생성함으로써 파이프라인을 단순화하고 변별력을 극대화했다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
SBT는 **Patch Embedding $\rightarrow$ Stacked EoC Blocks $\rightarrow$ Prediction Heads**의 순서로 구성된다.

### 1. Patch Embedding (PaE)
템플릿 이미지 $z$와 검색 이미지 $x$를 입력으로 받는다. 커널 크기 $7\times7$, 스트라이드 4인 합성곱 층 $\phi_p^0$과 Layer Normalization(LN)을 통해 초기 특징 맵 $f_z^0, f_x^0$를 생성한다.
$$f_z^0, f_x^0 = \text{LN}(\phi_p^0(z)), \text{LN}(\phi_p^0(x))$$

### 2. Extract-or-Correlation (EoC) Block
EoC 블록은 $\text{Self-Attention(SA)}$과 $\text{Cross-Attention(CA)}$을 모두 수행할 수 있는 핵심 단위이다. 연산 복잡도를 줄이기 위해 $\text{Spatial-Reduction Global attention (SRG)}$ 방식을 채택하여 Key($k$)와 Value($v$)의 공간 해상도를 낮춘 후 전역 어텐션을 계산한다.

- **공통 연산**: 쿼리($q$), 키($k$), 밸류($v$)를 선형 투영 $\omega$를 통해 생성한다.
  $$q_i = [\chi_q(f_i)]^T \omega_q, \quad k_i = [\chi_k(f_i)]^T \omega_k, \quad v_i = [\chi_v(f_i)]^T \omega_v \quad (i \in \{z, x\})$$
- **어텐션 계산**:
  $$\tilde{f}_{ij} = \text{Softmax}\left(\frac{q_i k_j^T}{\sqrt{d_h}}\right) v_j$$
- **EoC-SA (Self-Attention)**: 동일한 이미지 내에서 특징을 융합한다.
  $$f_z := f_z + \tilde{f}_{zz}, \quad f_x := f_x + \tilde{f}_{xx}$$
- **EoC-CA (Cross-Attention)**: 서로 다른 이미지 간의 특징을 교차 융합하여 상관관계를 모델링한다.
  $$f_z := f_z + \tilde{f}_{zx}, \quad f_x := f_x + \tilde{f}_{xz}$$

### 3. Position Encoding 및 직접 예측 (Direct Prediction)
위치 정보 제공을 위해 $3\times3$ depth-wise convolution을 사용하는 $\text{Conditional Position Encoding}$을 적용한다. 최종적으로 출력된 검색 이미지 특징 $\hat{f}_x$는 별도의 상관관계 연산 없이 분류 헤드($\Phi_{cls}$)와 회귀 헤드($\Phi_{reg}$)로 직접 전달되어 타겟의 위치와 크기를 예측한다. 이때 공간-채널 의존성을 동시에 모델링하는 $\text{Mix-MLP Blocks (MMB)}$가 사용된다.

### 4. 학습 절차 및 손실 함수
- **사전 학습**: ImageNet 데이터셋에서 분류 작업을 통해 4단계 SBT 모델을 사전 학습한다.
- **미세 조정**: 추적 데이터셋에서 $\text{Cross-Entropy Loss}$ (분류)와 $\text{GIoU Loss} + L_1 \text{ Loss}$ (회귀)를 사용하여 학습한다.

## 📊 Results

### 실험 설정
- **데이터셋**: GOT-10k, LaSOT, VOT2020, OTB100.
- **지표**: Average Overlap (AO), Success Rate (SR), Precision, EAO 등.
- **비교 대상**: STARK, TransT, DiMP, SiamRPN++ 등 최신 SOTA 추적기.

### 주요 결과
- **성능 우위**: GOT-10k 벤치마크에서 SBT-base 및 SBT-large 버전이 STARK, TransT, DiMP 등 기존 SOTA 모델보다 높은 AO를 기록하였다.
- **효율성**: SBT-small 및 SBT-light 버전은 모델 크기가 훨씬 작음에도 불구하고 경쟁력 있는 성능을 보였으며, 실시간 추적이 가능하다.
- **범용성 (CAT)**: SBT의 특징 네트워크를 기존 추적기(SiamFCpp, DiMP, STARK, STM)의 백본으로 교체한 $\text{Correlation-Aware Trackers (CAT)}$들이 모두 기존 베이스라인보다 성능이 향상됨을 확인하였다. 예를 들어, $\text{SiamFCpp-CA}$는 AO 기준 $\text{SiamFCpp}$보다 5.2%p 향상된 결과를 보였다.

## 🧠 Insights & Discussion

### 이론적 분석 및 강점
1.  **Translation Invariance**: 기존 CNN 기반 추적기는 패딩(Padding) 문제로 인해 평행 이동 불변성(Translation Invariance)을 유지하기 위해 복잡한 크롭핑이나 샘플링 전략을 썼다. 반면 SBT는 전역 수용 영역을 가진 EoC 블록과 순열 불변성(Permutation Invariance)을 가진 토큰 구조를 사용하여 이론적으로 평행 이동 불변성을 더 쉽게 확보한다.
2.  **Cross-Attention의 효과**: $\text{Cross-Attention}$은 수학적으로 두 개의 $\text{Dynamic Convolution (D-Conv)}$과 하나의 $\text{Softmax}$ 층으로 분해될 수 있다. 이는 기존의 depth-wise correlation 방식보다 두 배 더 효과적인 모델링 능력을 가짐을 의미한다.
3.  **타겟 의존적 임베딩**: T-SNE 시각화 결과, 네트워크가 깊어질수록 SBT의 검색 특징은 타겟(Green)과 방해물/배경(Blue/Pink)이 명확하게 분리되는 양상을 보였다. 이는 Siamese-like 네트워크가 타겟에 무관한(Target-unaware) 특징을 추출하는 것과 대조적이다.

### 한계 및 비판적 해석
- **폐색(Occlusion) 문제**: 타겟이 방해물에 의해 완전히 가려지거나 검색 영역을 완전히 벗어나는 경우, 쌍 기반(Pairwise) 추적 파이프라인의 한계로 인해 추적에 실패하는 모습이 관찰되었다.
- **연산 효율성**: 현대의 과학 계산 패키지들이 Transformer의 Attention 연산을 가속화하는 데 최적화되어 있지 않아, 실제 구현 상의 연산 효율성 문제가 있을 수 있음을 언급하였다.

## 📌 TL;DR

본 논문은 특징 추출 단계에서부터 템플릿과 검색 이미지의 상관관계를 깊게 통합한 **Single Branch Transformer (SBT)**를 제안하였다. 핵심은 $\text{Cross-Attention}$을 네트워크 층마다 배치하여 타겟에 의존적인 변별력 있는 특징을 생성하고, 이를 통해 별도의 상관관계 연산 단계 없이도 정밀한 추적이 가능하게 한 것이다. 이 연구는 SOTA 수준의 성능을 달성했을 뿐만 아니라, 다른 기존 추적기의 백본으로 적용했을 때도 성능을 높일 수 있음을 증명하여 향후 타겟 의존적 특징 추출 연구에 중요한 이정표를 제시하였다.