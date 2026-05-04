# An Empirical Study of Mamba-based Pedestrian Attribute Recognition

Xiao Wang, Weizhe Kong, Jiandong Jin, Shiao Wang, Ruichong Gao, Qingchuan Ma, Chenglong Li, Jin Tang (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 보행자 속성 인식(Pedestrian Attribute Recognition, PAR) 분야에서 기존 Transformer 기반 모델들이 가지는 높은 계산 복잡도 문제이다. Transformer의 Self-attention 메커니즘은 입력 시퀀스 길이 $N$에 대해 $O(N^2)$의 계산 복잡도를 가지므로, 학습 및 추론 과정에서 상당한 컴퓨팅 자원을 소모한다.

PAR은 보행자의 외형적 특징(예: 머리 색상, 모자 착용 여부, 가방 소지 여부 등)을 식별하는 작업으로, 이는 보행자 검출 및 재식별(Re-identification)과 같은 상위 시각 작업의 성능을 높이는 핵심적인 중간 단계의 시맨틱 표현을 제공한다. 따라서 본 연구의 목표는 Transformer 수준의 정확도를 유지하거나 능가하면서도, 계산 비용을 획기적으로 줄일 수 있는 선형 복잡도 $O(N)$의 Mamba(State Space Model, SSM) 아키텍처를 PAR 작업에 최적화하여 적용하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Mamba 아키텍처를 PAR 작업에 통합하고 그 효용성을 다각도로 검증한 실험적 연구에 있다. 주요 기여 사항은 다음과 같다.

1.  **두 가지 Mamba 기반 PAR 프레임워크 제안**: 순수 이미지 기반의 다중 레이블 분류(Multi-label Classification) 프레임워크와 이미지-텍스트 융합(Image-Text Fusion) 프레임워크를 설계하여 Mamba의 적용 가능성을 확인하였다.
2.  **하이브리드 Mamba-Transformer 구조 설계**: Mamba 단독 모델의 한계를 극복하기 위해 Transformer와 Mamba를 결합한 8가지 하이브리드 변형 구조를 제안하고, 최적의 결합 방식을 탐색하였다.
3.  **광범위한 벤치마크 검증**: PA100K, PETA, RAP-V1, RAP-V2, WIDER, PETA-ZS, RAP-ZS, MSP60K 등 총 8개의 데이터셋을 통해 제안 방법론의 유효성과 효율성을 입증하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계
- **CNN 기반**: 초기 PAR 모델들은 CNN을 활용한 다중 작업 학습(Multi-task Learning)을 통해 특징을 추출하였으나, 속성 간의 상관관계를 충분히 학습하는 데 한계가 있었다.
- **RNN/GNN 기반**: 속성 간의 관계를 시퀀스 예측 문제로 정의하고 LSTM 등을 사용하여 문맥적 관계를 캡처하려 하였다.
- **Transformer 기반**: 전역 주의 집중(Global Attention) 메커니즘을 통해 더 정교한 특징 추출이 가능해졌으며, 최근에는 텍스트 정보를 결합한 멀티모달 융합 방식이 주류를 이루고 있다. 그러나 앞서 언급한 대로 $O(N^2)$의 복잡도로 인한 효율성 문제가 심각하다.

### 차별점
본 논문은 기존의 $O(N^2)$ 복잡도를 가진 Transformer 대신, 선형 복잡도를 가지는 State Space Model(SSM) 기반의 Mamba를 도입하였다. 특히 단순히 백본을 교체하는 것에 그치지 않고, Mamba와 Transformer의 장점을 결합한 하이브리드 구조를 제안함으로써 효율성과 정확도 사이의 최적의 균형점을 찾으려 했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. State Space Model (SSM) 기초
Mamba의 기반이 되는 SSM은 1차원 시퀀스를 $N$차원 은닉 상태로 변환하는 선형 필터링 구조이다. 연속 시간 SSM의 수식은 다음과 같다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

여기서 $x(t)$는 입력 시퀀스, $h'(t)$는 은닉 상태의 도함수, $A$는 상태 행렬, $B$는 입력 행렬, $C$는 출력 행렬, $D$는 피드스루 행렬이다. 이를 딥러닝에 적용하기 위해 Zero Order Hold(ZOH) 등의 방법으로 이산화(Discretization)하며, timescale 파라미터 $\Delta$를 도입하여 이산 파라미터 $\bar{A}, \bar{B}, \bar{C}$를 얻는다.

### 2. Mamba 기반 PAR 프레임워크
본 논문은 두 가지 경로를 제안한다.

- **순수 이미지 기반 경로**: VMamba 또는 Vim 백본을 사용하여 이미지를 패치 토큰으로 변환하고, Mamba 블록을 통해 시각 특징 $X_v$를 추출한 뒤 분류 헤드로 전달한다.
- **이미지-텍스트 융합 경로**:
    - **시각 인코더**: Vim 또는 VMamba를 통해 시각 토큰 $X_v$를 추출한다.
    - **텍스트 인코더**: BERT 토크나이저와 Text Mamba 블록을 통해 속성 레이블(예: "Black hair")을 시맨틱 토큰 $X_s$로 변환한다.
    - **VSF Mamba (Visual-Semantic Fusion)**: 시각 토큰과 시맨틱 토큰을 결합하여 $[X_v, X_s]$ 형태의 입력을 구성하고, Mamba 블록 스택을 통해 두 모달리티 간의 상호작용을 모델링한다.

### 3. 학습 목표 및 손실 함수
최종 예측은 Feed-Forward Network(FFN)와 Sigmoid 함수를 통해 각 속성의 확률값 $P$를 도출한다.

$$P = \text{Sigmoid}(\text{FFN}(X))$$

클래스 불균형 문제를 해결하기 위해 가중치 교차 엔트로피 손실 함수(Weighted Cross-Entropy Loss)를 사용한다.

$$L = -\sum_{j=1}^{L} w_j (y_j \log(p_j) + (1-y_j) \log(1-p_j))$$

여기서 $w_j$는 학습 세트 내 해당 속성의 양성 샘플 비율과 양의 상관관계를 가지는 가중치이다.

### 4. 하이브리드 Mamba-Transformer 구조
Mamba 단독 모델의 성능을 높이기 위해 8가지 변형 구조를 설계하였다.
- **PaFusion (Parallel Fusion)**: ViT-B와 Vim-S를 병렬로 연결하여 특징을 합산한다.
- **N-ASF / ASF (Serial Fusion)**: Vim과 ViT를 직렬로 연결하며, N-ASF는 단순 연결, ASF는 교차 연결 방식을 취한다.
- **MaFormer**: Vim-S를 보조 특징 추출기로 사용하여 ViT의 입력을 강화한다.
- **MaHDFT (Hierarchical Dense Fusion)**: 동결된 ViT-B의 모든 레이어 출력을 Vim-S가 통합 처리하는 구조이다. 본 논문에서 가장 우수한 성능을 보였다.
- **KDTM / MaKDF**: Teacher network(ViT)에서 Student network(Vim)로 지식 증류(Knowledge Distillation)를 수행하는 방식이다.

## 📊 Results

### 실험 설정
- **데이터셋**: PA100K, PETA, RAP-V1/V2, WIDER, PETA-ZS, RAP-ZS, MSP60K.
- **평가 지표**: mA (mean Average Precision), Accuracy, Precision, Recall, F1-measure.
- **백본**: VMamba-B (768-D), Vim-S (384-D), ViT-B/S.

### 주요 결과
1.  **순수 Mamba 성능**: VMamba-B 기반 모델이 대부분의 데이터셋에서 가장 우수한 성능을 보였으며, 특히 PETA와 RAP-V2 데이터셋에서는 Transformer 기반의 VTB* 모델과 비슷하거나 능가하는 결과를 얻었다.
2.  **모달리티 융합의 영향**: Vim-S의 경우 텍스트 정보를 융합했을 때 성능이 향상되었으나, VMamba-B의 경우 오히려 성능이 하락하는 경향을 보였다. 이는 VMamba의 강력한 표현 능력이 단순한 텍스트 융합 과정에서 오히려 제약을 받을 수 있음을 시사한다.
3.  **하이브리드 모델의 우위**: MaHDFT 구조가 가장 높은 성능을 기록하며, 심지어 가장 거대한 ViT 모델보다도 뛰어난 결과를 보였다.
4.  **Zero-shot 성능**: RAP-ZS 데이터셋에서 Mamba 기반 모델이 VTB(ViT-B/16)보다 더 높은 mA를 기록하여, Mamba가 보행자 속성 인식에 있어 높은 적응력을 가짐을 확인하였다.

### 효율성 분석
- **파라미터 수**: Mamba 기반 모델들이 ViT 모델들에 비해 파라미터 수가 적어 모델의 경량화 및 파인튜닝에 유리하다.
- **추론 속도 및 메모리**: 예상과 달리, 제한된 이미지 해상도 환경에서 VMamba-B는 ViT-B보다 추론 시간이 더 걸리고 GPU 메모리 사용량이 약 2배가량 높았다.

## 🧠 Insights & Discussion

### 강점 및 발견
- **Mamba의 잠재력**: 본 연구는 Mamba가 PAR 작업에서 Transformer를 대체하거나 보완할 수 있는 강력한 후보임을 입증하였다. 특히 파라미터 효율성이 뛰어나다.
- **구조적 최적화의 중요성**: 단순히 백본을 교체하는 것보다, MaHDFT와 같이 Transformer의 계층적 특징을 Mamba로 통합하는 하이브리드 방식이 훨씬 효과적이다.

### 한계 및 비판적 해석
- **효율성 역설**: Mamba의 이론적 복잡도는 $O(N)$이지만, 실제 구현체(특히 VMamba)에서는 이미지 해상도가 낮을 때 오히려 Transformer보다 메모리 사용량이 높고 속도가 느린 현상이 나타났다. 이는 현재의 Mamba 구현 최적화가 고해상도나 매우 긴 시퀀스에 최적화되어 있어, PAR과 같은 소규모 입력 작업에서는 이점이 적기 때문으로 해석된다.
- **멀티모달 융합의 미흡**: 텍스트-이미지 융합 결과가 백본 모델(Vim vs VMamba)에 따라 상이하게 나타났으며, 일관된 성능 향상 방안을 제시하지 못했다. 저자 역시 이 부분을 한계로 명시하였다.

## 📌 TL;DR

본 논문은 계산 비용이 높은 Transformer 기반의 보행자 속성 인식(PAR) 작업을 해결하기 위해 선형 복잡도를 가진 **Mamba 아키텍처를 도입하고 분석한 실험적 연구**이다. 연구 결과, **VMamba-B** 백본이 단독 모델 중 가장 우수했으며, 특히 Transformer의 계층적 특징을 Mamba로 융합한 **MaHDFT 하이브리드 구조**가 최상의 성능을 기록하였다. Mamba는 파라미터 수 측면에서 효율적이며 Zero-shot 상황에서도 강점을 보였으나, 실제 추론 속도와 메모리 효율은 입력 해상도가 낮은 PAR 작업 특성상 Transformer 대비 이점이 크지 않았다. 이 연구는 향후 Mamba 기반의 효율적인 멀티모달 융합 구조 설계에 중요한 가이드라인을 제공한다.