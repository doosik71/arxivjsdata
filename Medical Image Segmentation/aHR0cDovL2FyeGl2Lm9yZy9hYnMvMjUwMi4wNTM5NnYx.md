# A Novel Convolutional-Free Method for 3D Medical Imaging Segmentation

Canxuan Gang(2025)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상 분할(Segmentation)에서 기존 Convolutional Neural Networks(CNNs)가 가진 구조적 한계와 데이터 획득의 불균형 문제를 해결하고자 한다.

첫째, CNN은 지역적 수용장(Local Receptive Field) 특성으로 인해 장거리 의존성(Long-range dependencies)과 전역적 문맥(Global context)을 포착하는 데 어려움이 있으며, 이는 특히 정밀하고 복잡한 의료 구조물을 분할할 때 성능 저하의 원인이 된다.

둘째, CT 영상 데이터에서 두꺼운 슬라이스(Thick slices)와 얇은 슬라이스(Thin slices) 사이의 도메인 차이 문제가 존재한다. 얇은 슬라이스는 해상도가 높고 등방성(Isomorphic)을 가져 정밀한 3D 볼륨 렌더링이 가능하지만, 전문의에 의한 픽셀 단위 정답(Ground Truth) 데이터가 매우 부족하다. 반면 두꺼운 슬라이스는 데이터가 많지만 해상도가 낮아 정밀도가 떨어진다.

따라서 본 연구의 목표는 완전한 Convolution-free 아키텍처를 설계하여 전역 문맥 파악 능력을 높이고, 두꺼운 슬라이스의 어노테이션을 활용해 얇은 슬라이스를 정밀하게 분할할 수 있는 도메인 적응(Domain Adaptation) 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같은 세 가지 설계 아이디어로 요약된다.

1.  **완전한 Convolution-free 아키텍처**: 기존의 Hybrid 모델(CNN-Transformer 결합형)에서 벗어나, Transformer와 Self-attention 메커니즘만을 사용한 순수 비-합성곱 구조를 제안하여 3D 의료 영상의 전역적 특징 추출 능력을 극대화하였다.
2.  **Thin-Thick Adaptation을 위한 Joint Loss 설계**: 두꺼운 슬라이스의 어노테이션을 통해 얇은 슬라이스의 분할 성능을 높이는 공동 가중 손실 함수(Joint weighted loss)를 제안하였다. 이를 통해 데이터 부족 문제를 해결하고 고해상도 얇은 슬라이스 영상에 대한 정밀한 마스크를 생성한다.
3.  **새로운 벤치마크 데이터셋 제공**: 뇌출혈(Brain Hemorrhage)의 5가지 세부 클래스(EDH, ICH, IVH, SAH, SDH)를 포함하는 얇은 슬라이스 기반의 다중 세만틱 분할(Multi-semantic segmentation) 데이터셋을 구축하고 이를 공개하여 후속 연구의 기반을 마련하였다.

## 📎 Related Works

논문에서는 3D 의료 영상 분할의 발전 과정을 다음과 같이 설명한다.

*   **CNN 기반 모델**: 3D U-Net과 nnU-Net이 대표적이며, 특히 nnU-Net은 전처리와 후처리 파이프라인의 최적화를 통해 현재까지도 SOTA(State-of-the-art) 성능을 보이는 강력한 모델로 평가받는다. 하지만 이들은 전역 문맥 파악 능력이 부족하다는 한계가 있다.
*   **Hybrid 모델**: TransUNet, CoTr, nnFormer 등은 CNN의 특징 추출 능력과 Transformer의 전역 문맥 파악 능력을 결합하였다. 그러나 이들은 여전히 CNN 구조에 의존하고 있어 완전한 Convolution-free 모델이라고 할 수 없다.
*   **Convolution-free 모델**: Karimi et al.의 연구가 있었으나, 얇은 슬라이스 분할을 위한 도메인 적응 기능이 부재하며 코드와 데이터셋의 공개 부족으로 인해 검증 가능성이 떨어진다는 한계가 명시되었다.
*   **최신 동향**: 최근 Mamba와 같은 State-space models가 효율적인 시퀀스 모델링과 장거리 의존성 파악 능력을 보여주며 새로운 대안으로 떠오르고 있다.

본 제안 방법은 기존 Hybrid 모델들의 타협안에서 벗어나 완전한 Transformer 기반 구조를 채택하고, 특히 의료 현장의 실질적 문제인 슬라이스 두께 간의 데이터 불균형 문제를 직접적으로 해결하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

제안하는 시스템은 **Extract Block $\rightarrow$ 3D Patch Embeddings $\rightarrow$ 3D Patch Encodings $\rightarrow$ Transformer Encoder $\rightarrow$ MLP Decoder** 순으로 구성된 파이프라인을 가진다.

### 1. 데이터 처리 및 임베딩
입력 3D 영상(채널, 높이, 너비, 깊이)은 먼저 크기 $W \times W \times W \times c$의 블록 $B$로 추출된다.
$$B \in \mathbb{R}^{W \times W \times W \times c}$$
이 블록 $B$는 겹치지 않는 $N=n^3$개의 3D 패치 $\{p_i\}$로 분할된다. 각 패치는 벡터로 평탄화(Flatten)된 후 학습 가능한 임베딩 층 $E \in \mathbb{R}^{D \times w^3c}$를 통과하여 임베딩 표현 $E^P \in \mathbb{R}^D$를 얻는다.

### 2. 패치 인코딩 및 Transformer Encoder
위치 정보를 제공하기 위해 $\sin, \cos$ 함수 기반의 위치 인코딩 행렬 $E_{pos} \in \mathbb{R}^{D \times N}$를 생성하여 임베딩 결과에 더한다.
$$X_0 = [E^P_1; \dots; E^P_N] + E_{pos}$$
이후 $k$개의 동일한 층으로 구성된 Transformer Encoder를 통과한다. 각 층은 다음 두 가지 서브 레이어로 구성된다.

*   **Multi-head Self-Attention (MHSA)**: 쿼리($Q$), 키($K$), 밸류($V$) 행렬을 사용하여 어텐션을 계산한다.
    $$A(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D_h}}\right)V$$
    최종 결과는 여러 헤드의 출력을 연결(Concat)하고 가중치 행렬 $W^O$를 곱하여 산출한다.
*   **Feed-Forward Neural Network (FFNN)**: ReLU 활성화 함수를 포함한 두 번의 선형 변환으로 구성된다.
    $$\text{FFNN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

각 서브 레이어는 잔차 연결(Residual connection)과 레이어 정규화(Layer Normalization)를 거친다.

### 3. MLP Decoder 및 출력
Encoder의 출력 $X_k$는 선형 변환 층을 통해 채널 차원이 조정되며, 블록 $B$의 중심 패치에 대해 Softmax 함수를 적용하여 최종 마스크를 예측한다.
$$\text{Mask} = \text{softmax}(X_k W + b)$$

### 4. Thin-thick Adaptation (도메인 적응)
얇은 슬라이스 분할을 위해 두꺼운 슬라이스의 어노테이션을 활용하는 Joint Weighted Loss를 설계하였다. 얇은 슬라이스에서 평균 강도 투영(AIP)을 통해 두꺼운 슬라이스를 생성하고, 다음 세 가지 제약 조건을 손실 함수에 반영한다.
1.  두꺼운 슬라이스에 대한 예측 마스크가 실제 Ground Truth와 일치해야 한다.
2.  대응하는 얇은 슬라이스들의 마스크 강도 평균이 두꺼운 슬라이스의 Ground Truth와 일치해야 한다.
3.  대응하는 얇은 슬라이스들의 특징 맵(Feature maps) 평균이 두꺼운 슬라이스의 특징 맵과 일치해야 한다.

## 📊 Results

본 논문에서는 구체적인 실험 수치보다는 실험 설계 및 평가 기준을 상세히 제시하고 있다.

*   **데이터셋**: 공공 데이터셋(Medical Segmentation Decathlon 등 두꺼운 슬라이스 중심)과 자체 구축한 뇌출혈 얇은 슬라이스 데이터셋을 사용한다.
*   **평가 지표**:
    *   **mIoU (mean Intersection over Union)**: 클래스별 예측 영역과 정답 영역의 교집합을 합집합으로 나눈 값의 평균을 측정한다.
        $$\text{IoU} = \frac{\text{GroundTruth} \cap \text{Prediction}}{\text{GroundTruth} \cup \text{Prediction}}$$
    *   **DSC (Dice Similarity Coefficient)**: 예측과 정답 간의 겹침 정도를 측정하며, 클래스 불균형이 심한 의료 영상에서 유용하다.
        $$\text{DSC} = \frac{2 \times \text{Intersection}}{\text{Union}}$$
*   **실험 계획**:
    1.  **비교 실험**: 제안 모델과 기존 3D 분할 모델들(CNN, Hybrid 등)을 두꺼운 슬라이스 및 얇은 슬라이스 환경에서 각각 비교 분석한다.
    2.  **절제 실험(Ablation Study)**: 모델의 각 구성 요소가 최종 성능에 미치는 기여도를 개별적으로 분석한다.

## 🧠 Insights & Discussion

논문은 제안 방법의 강점과 함께 향후 해결해야 할 잠재적 문제점들을 심도 있게 논의한다.

**강점** 및 **기대 효과**:
Transformer 구조를 통해 전역 문맥을 효과적으로 포착함으로써 더 정확한 세만틱 레이블링이 가능하며, 특히 Thin-thick adaptation을 통해 수술 계획 및 내비게이션에 필수적인 고해상도 등방성 마스크를 생성할 수 있다는 점이 강력한 이점으로 작용한다.

**한계 및 논의 사항**:
1.  **Long-tail 문제**: 의료 데이터 특성상 희귀 질환 클래스의 샘플이 부족한 불균형 문제가 발생한다. 저자는 이를 해결하기 위해 가중 손실 함수(Weighted loss)나 Diffusion 모델(예: DiffuMask)을 이용한 합성 데이터 생성을 대안으로 제시한다.
2.  **사전 학습 모델의 부족**: 일반 컴퓨터 비전과 달리 3D 의료 영상 분야는 적절한 사전 학습 모델이 부족하다. 이를 극복하기 위해 MoCo나 MAE 같은 대규모 자기지도 학습(Self-supervised learning) 기반의 인코더를 구축하여 특징 추출기로 활용할 가능성을 언급한다.
3.  **윤리 및 보안 문제**: 의료 데이터의 민감성으로 인해 데이터 수집 및 공유 과정에서 환자 개인정보 보호와 익명화 처리가 필수적이며, 이에 대한 엄격한 가이드라인 준수가 필요함을 강조한다.

## 📌 TL;DR

본 논문은 3D 의료 영상 분할에서 CNN의 지역적 한계를 극복하기 위해 **완전한 Convolution-free Transformer 아키텍처**를 제안한다. 특히 **두꺼운 슬라이스의 어노테이션을 활용해 얇은 슬라이스를 학습시키는 Joint Loss**를 통해 데이터 부족 문제를 해결하고 분할 정밀도를 높였다. 또한, 공개되지 않았던 **뇌출혈 얇은 슬라이스 다중 세만틱 데이터셋을 벤치마크로 제공**하여, 향후 고해상도 의료 영상 분할 연구 및 정밀 수술 계획 수립에 기여할 가능성이 높다.