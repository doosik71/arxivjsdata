# FouRA: Fourier Low Rank Adaptation

Shubhankar Borse, Shreya Kadambi, Nilesh Prasad Pandey, et al. (2024)

## 🧩 Problem to Solve

본 논문은 대규모 모델을 효율적으로 미세 조정하기 위해 널리 사용되는 Low-Rank Adaptation (LoRA) 방식이 텍스트-이미지 확산 모델(Text-to-Image Diffusion Models)에 적용될 때 발생하는 한계점을 해결하고자 한다.

가장 핵심적인 문제는 LoRA로 미세 조정된 모델이 학습 데이터의 샘플을 그대로 복제하려는 경향을 보이며, 이로 인해 생성된 이미지의 다양성이 부족해지는 **Distribution Collapse**(분포 붕괴) 현상이 발생한다는 점이다. 이러한 현상은 어댑터의 강도($\alpha$)가 높거나, 작은 데이터셋으로 학습된 고순위(high rank) 어댑터를 사용할 때 더욱 심화된다. 또한, LoRA의 랭크(rank) 설정은 매우 민감한 파라미터로, 랭크가 너무 낮으면 언더피팅(underfitting)이 발생하고 너무 높으면 오버피팅(overfitting)으로 인한 데이터 복제 아티팩트가 발생한다.

따라서 본 연구의 목표는 데이터 복제 문제를 해결하고 생성 이미지의 다양성을 확보하며, 입력 데이터 및 상황에 따라 유연하게 랭크를 조절할 수 있는 새로운 저순위 적응(low-rank adaptation) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 저순위 투영(low-rank projection)을 특징 공간(feature space)이 아닌 **주파수 영역(Frequency Domain)**에서 수행하는 것이다.

1. **FouRA 제안**: 특징 공간의 픽셀 또는 채널 차원에 대해 푸리에 변환을 적용하고, 주파수 영역에서 down-projection과 up-projection을 수행하는 최초의 저순위 어댑터 모듈을 제안한다.
2. **Adaptive Rank Gating**: 주파수 영역 내에서 학습 가능한 마스킹 전략을 도입하여, 각 레이어의 유효 랭크(effective rank)를 입력값과 확산 공정(diffusion process)의 타임스텝에 따라 동적으로 변경함으로써 일반화 성능을 높인다.
3. **Decorrelated Orthonormal Basis**: 주파수 영역에서의 학습이 서로 상관관계가 낮은 직교 기저(orthogonal basis)를 형성함을 입증하였다. 이를 통해 별도의 공동 학습(joint training) 없이도 여러 스타일이나 개념의 어댑터를 효과적으로 병합(merge)할 수 있게 한다.
4. **범용성 입증**: 시각적 작업뿐만 아니라 언어 모델 작업(GLUE benchmark)에서도 LoRA 및 기존 적응형 랭크 방식보다 우수한 성능을 보임을 확인하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 검토하고 차별점을 제시한다.

- **Text-to-Image Diffusion Models**: Stable Diffusion과 같은 모델들은 LoRA를 통해 빠르게 새로운 스타일을 학습할 수 있으나, 앞서 언급한 다양성 부족 문제가 존재한다.
- **Fourier Transforms in Generative Literature**: 푸리에 연산자가 연속적인 표현을 제공하고 전역 컨볼루션(global convolution) 역할을 수행할 수 있다는 연구들이 있었으나, 이를 저순위 적응 공간(low-rank space)에 적용한 시도는 이전까지 없었다.
- **Low Rank Adaptation (LoRA)**: 기존의 LoRA, SVDiff, AdaLORA 등이 가중치 행렬의 부분 공간을 제약함으로써 효율성을 꾀했다. 특히 SoRA와 같은 적응형 랭크 방식이 제안되었으나, 이는 주로 학습 중에 결정되며 추론 시에는 고정된다는 한계가 있다.
- **Adapter Merging**: 여러 LoRA를 병합하려는 시도(MoLE, ZipLoRA 등)가 있었으나, 대개 복잡한 공동 학습 과정이 필요했다. FouRA는 주파수 영역의 특성을 이용해 학습 없이도(training-free) 유연한 병합이 가능하다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

FouRA는 입력 특징을 주파수 영역으로 변환한 뒤 저순위 투영을 수행하고, 다시 원래의 공간으로 되돌리는 구조를 가진다. 기본 LoRA가 특징 공간에서 $BAx$를 수행하는 것과 달리, FouRA는 주파수 변환 $\mathcal{F}(\cdot)$와 역변환 $\mathcal{F}^{-1}(\cdot)$ 사이에 저순위 행렬들을 배치한다.

### 2. 주파수 영역에서의 저순위 적응

입력 특징 $z_{in}$에 대해 주파수 영역에서의 적응 결과는 다음과 같은 방정식으로 표현된다.

$$z_{out} = z_{og} + z_{foura} = W_0 z_{in} + \mathcal{F}^{-1}(B \alpha G(z_{lr}) \cdot A \mathcal{F}(z_{in}))$$

여기서 $z_{og}$는 기본 모델의 출력이며, $\mathcal{F}$와 $\mathcal{F}^{-1}$는 각각 정규화된 전방 및 역 푸리에 변환을 의미한다. $A$는 down-projection, $B$는 up-projection 행렬이며, $\alpha$는 어댑터 강도를 조절하는 스칼라 값이다.

### 3. 주파수 변환 (Frequency Transforms)

본 연구에서는 이산 푸리에 변환(DFT)과 이산 코사인 변환(DCT)을 모두 조사하였다. 1D DFT를 임베딩 차원 $k_1$에 적용하여 각 토큰의 주파수 성분을 추출한다. 특히 DCT의 경우, 신호를 두 배 길이로 확장하여 대칭적으로 배치한 후 DFT를 계산함으로써 더 매끄러운 표현과 낮은 오버피팅 효과를 얻었다.

### 4. Adaptive Rank Gating Method

FouRA의 핵심인 적응형 게이팅 메커니즘 $G(\cdot)$는 저순위 부분 공간 내에서 특정 주파수 기저를 동적으로 제거하여 유효 랭크를 조절한다.

$$G(z_{lr}) = \begin{cases} 1, & \text{if } S(H(G z_{lr})) == 1 \\ 0, & \text{otherwise} \end{cases}$$

여기서 $z_{lr} = A \mathcal{F}(z_{in})$이며, $H(\cdot)$는 엔트로피 함수, $S(\cdot)$는 시그모이드 함수, $G$는 학습 가능한 MLP(Multi-Layer Perceptron)이다. 이 게이팅 함수는 입력 데이터와 확산 타임스텝에 따라 실시간으로 변화하므로, 추론 시에도 유연하게 랭크를 조절할 수 있다.

### 5. 다중 어댑터 병합 (Combining Multiple Adapters)

FouRA는 주파수 영역에서 학습된 부분 공간들이 서로 디코릴레이션(decorrelation) 되어 있어, 단순히 출력값을 선형 결합하는 것만으로도 여러 스타일을 효과적으로 융합할 수 있다. 특히 Concept Sliders 작업에서는 서로 다른 개념의 어댑터들을 $\epsilon$-공간에서 합성하여 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 스타일 전이(Bluefire, Paintings, 3D, Origami), 개념 편집(Age, Hair, Surprise), 언어 작업(GLUE benchmark)을 사용하였다.
- **모델**: Stable Diffusion v1.5, Realistic Vision v3.0, RoBERTa-Base를 사용하였다.
- **지표**: 이미지 품질 및 프롬프트 정렬도를 측정하는 **HPSv2.1**과 이미지 간 다양성을 측정하는 **LPIPS Diversity**를 주요 지표로 사용하였다.

### 2. 주요 정량적 결과

- **스타일 생성**: Table 2에 따르면, FouRA는 LoRA보다 LPIPS Diversity와 HPSv2 score 모두에서 유의미하게 높은 성능을 보였다. 특히 어댑터 강도 $\alpha$가 높을 때 LoRA는 다양성이 급격히 떨어지지만, FouRA는 높은 다양성을 유지하였다.
- **어댑터 병합**: Table 1에서 두 가지 스타일(Blue Fire, Paintings)을 병합했을 때, FouRA는 $\alpha$ 값이 높음에도 불구하고 HPSv2 score에서 LoRA 대비 최대 3%의 성능 향상을 보였으며, 시각적으로도 두 스타일의 특성을 모두 잘 보존하였다.
- **언어 모델 작업**: GLUE 벤치마크의 6개 작업 중 4개 작업에서 FouRA가 SoRA 및 일반 LoRA보다 우수한 성능을 기록하였다.

### 3. 정성적 결과 및 분석

- **데이터 복제 방지**: LoRA는 고강도 설정에서 동일한 이미지를 반복 생성하는 경향이 있으나, FouRA는 다양한 구도와 형태의 이미지를 생성하였다.
- **개념 편집**: '나이(Age)'나 '헤어스타일(Hair)'을 변경하는 Concept Slider 실험에서, LoRA는 강도가 높아지면 성별이 바뀌거나 얼굴 구조가 왜곡되는 현상이 발생했으나, FouRA는 원래 인물의 특징을 유지하면서 자연스럽게 속성만 변경하였다.

## 🧠 Insights & Discussion

### 1. 이론적 근거 및 강점

논문은 FouRA의 우수성을 수학적으로 분석한다.

- **SVD 분석**: FouRA의 가중치 행렬은 LoRA보다 특이값 분포(singular value spread)가 더 조밀하다(compact). 이는 동일 랭크에서 재구성 오차(reconstruction error)가 더 적음을 의미하며, 결과적으로 더 나은 일반화 성능으로 이어진다.
- **일반화 오차 상한**: 유효 랭크를 동적으로 조절함으로써 오버피팅과 언더피팅 사이의 최적의 균형점을 찾을 수 있으며, 주파수 영역의 조밀한 특이값 분포가 일반화 오차의 상한을 더 안정적으로 낮춘다는 점을 증명하였다.
- **부분 공간 분석**: FouRA가 학습한 부분 공간은 기본 모델의 가중치와 더 많이 디코릴레이션(decorrelation) 되어 있어, 치명적 망각(catastrophic forgetting) 없이 새로운 작업을 학습할 수 있다.

### 2. 한계점 및 논의

- **하드웨어 최적화**: 현재의 딥러닝 하드웨어는 행렬 곱셈(GEMM)과 컨볼루션에 최적화되어 있어, 푸리에 변환 연산이 상대적으로 덜 효율적일 수 있다는 점이 한계로 지적되었다. 하지만 최근 푸리에 연산자 관련 연구들이 증가하고 있어 이는 극복 가능한 문제로 보인다.
- **추론 오버헤드**: 적응형 마스킹을 적용할 때 LoRA 대비 약 0.02%의 연산 오버헤드가 발생하지만, 마스크를 고정(frozen mask)할 경우 오버헤드를 획기적으로 줄이면서도 LoRA보다 높은 성능을 유지할 수 있음을 확인하였다.

## 📌 TL;DR

FouRA는 LoRA의 학습 과정을 **주파수 영역(Frequency Domain)**으로 옮기고, **입력 의존적인 적응형 랭크 게이팅(Adaptive Rank Gating)**을 도입한 새로운 PEFT 기법이다. 이를 통해 기존 LoRA의 고질적인 문제였던 데이터 복제(data copying)와 분포 붕괴(distribution collapse)를 해결하여 생성 이미지의 다양성과 품질을 동시에 높였다. 특히 주파수 영역의 직교 기저 특성 덕분에 여러 어댑터를 학습 없이도 정교하게 병합할 수 있으며, 시각적 작업뿐만 아니라 언어 모델에서도 효과적임을 입증하였다. 이 연구는 향후 모델의 정교한 편집과 효율적인 다중 개념 융합 연구에 중요한 기여를 할 것으로 보인다.
