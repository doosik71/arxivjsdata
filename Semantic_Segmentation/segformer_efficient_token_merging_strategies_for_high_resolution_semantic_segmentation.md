# Segformer++: Efficient Token-Merging Strategies for High-Resolution Semantic Segmentation

Daniel Kienzle, Marco Kantonis, Robin Schön, Rainer Lienhart (2024)

## 🧩 Problem to Solve

본 논문은 고해상도 이미지의 시맨틱 세그멘테이션(Semantic Segmentation) 작업에서 Vision Transformer(ViT) 아키텍처가 겪는 계산 복잡도 문제를 해결하고자 한다. Transformer의 Self-attention 메커니즘은 입력 토큰 수의 제곱에 비례하는 $O(N^2)$의 계산 복잡도를 가지며, 이는 고해상도 이미지 처리 시 메모리 사용량 급증과 추론 속도 저하로 이어진다.

일반적으로는 입력 이미지의 해상도를 크게 낮추는 다운샘플링(Downsampling)을 통해 이를 해결하지만, 시맨틱 세그멘테이션, 단안 깊이 추정(Monocular Depth Estimation), 인간 포즈 추정(Human Pose Estimation)과 같은 dense pixel task에서는 세밀한 디테일이 중요하므로 과도한 다운샘플링은 성능 저하를 초래한다. 따라서 본 연구의 목표는 모델의 성능(mIoU 등)을 최대한 유지하면서 고해상도 이미지 처리 속도를 획기적으로 높일 수 있는 효율적인 토큰 감소 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 분류(Image Classification) 분야에서 제안된 **Token Merging** 전략을 Segformer 아키텍처에 맞게 최적화하여 적용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Segformer++ 아키텍처 제안**: 기존 Segformer의 계층적 구조와 컨볼루션 레이어의 특성을 유지하면서, Attention 계산 직전에 유사한 토큰을 병합하고 계산 후 다시 복원하는 'Merge-Attention-Unmerge' 파이프라인을 설계하였다.
2. **효율적인 토큰 병합 전략**: 유사도 기반의 Token Merging뿐만 아니라, 단순한 2D 평균 풀링을 이용한 **2D Neighbor Merging** 전략을 함께 제시하여 성능과 속도 간의 트레이드오프를 분석하였다.
3. **재학습 없는 가속화**: 제안된 방법론은 이미 학습된 모델의 가중치를 그대로 사용할 수 있어, 추가적인 재학습(Re-training) 없이도 추론 속도를 비약적으로 향상시킬 수 있음을 입증하였다.
4. **범용적 적용 가능성**: 시맨틱 세그멘테이션뿐만 아니라 인간 포즈 추정과 같은 다른 dense pixel task에서도 효과적임을 증명하였다.

## 📎 Related Works

기존의 dense pixel task 연구들은 주로 피라미드 구조(Pyramid Structure)나 다중 분기(Multiple Branches)를 통해 해상도와 계산 비용 사이의 균형을 맞추려 하였다. Transformer 기반 모델의 경우, Swin Transformer와 같이 윈도우 기반 attention을 사용하여 복잡도를 낮추려 했으나, 이는 전역 수용 영역(Global Receptive Field)을 상실한다는 단점이 있다.

Segformer는 Spatial Reduction Attention(SRA)을 도입하여 전역 수용 영역을 유지하면서 계산량을 줄였으나, 여전히 고해상도 이미지에서는 한계가 있다. 토큰 수를 줄이기 위한 기존의 접근 방식으로는 중요도가 낮은 토큰을 제거하는 **Token Pruning**이 있었으나, 이는 정보 손실을 초래하며 종종 모델의 재학습을 요구한다. 반면, **Token Merging**은 유사한 토큰을 결합함으로써 정보 손실을 최소화하고 재학습 없이 적용 가능하다는 차별점이 있다.

## 🛠️ Methodology

### 1. Segformer 기본 구조

Segformer는 MixTransformer(MiT) 인코더와 가벼운 컨볼루션 디코더로 구성된다. MiT는 $7 \times 7$의 겹치는 패치를 사용하여 세밀한 정보를 캡처하며, 4단계의 피라미드 구조를 통해 다중 스케일 특징 맵을 생성한다. 특히 **Spatial Reduction Attention(SRA)**은 Key($K$)와 Value($V$)에 스트라이드 $R$을 가진 2D 컨볼루션을 적용하여 토큰 수를 $\frac{1}{R^2}$로 줄임으로써 계산 복잡도를 $O(\frac{N^2}{R^2}D)$로 낮춘다.

### 2. Token Merging 전략

본 논문에서 적용한 Token Merging의 핵심은 유사한 토큰을 병합하여 전체 토큰 수 $N$을 줄이는 것이다. 유사도 측정에는 Bipartite Soft Matching이 사용되며, 두 그룹 $A, B$ 사이의 유사도는 다음과 같이 계산된다.
$$\text{similarity}(A, B) = A \cdot B^T$$
유사도가 가장 높은 토큰들을 병합하며, 병합된 토큰은 평균값으로 대표된다.

### 3. Segformer++ 아키텍처

Segformer는 컨볼루션 레이어를 빈번하게 사용하므로, 토큰을 단순히 병합한 채로 유지하면 2D 구조가 파괴되어 컨볼루션 연산이 불가능해진다. 이를 해결하기 위해 Segformer++는 다음과 같은 절차를 따른다.

- **Merge**: Attention 계산 직전에 유사도 기반으로 토큰을 병합한다.
- **Attention**: 감소된 토큰 수로 Attention을 수행한다.
- **Unmerge**: Attention 결과값을 다시 원래의 위치로 복사하여 2D 구조를 복원한다.

이 과정은 SRA와 결합되어 적용된다. 또한, 각 스테이지 $i$마다 쿼리($Q$)와 키-값($KV$)에 대해 서로 다른 감소율 $r^q_i$와 $r^{(kv)}_i$를 적용하여 최적의 성능-속도 균형을 찾는다. 여기서 감소율 $r$에 따른 토큰 감소 배수 $\lambda$는 다음과 같다.
$$\lambda = \frac{1}{1-r}$$

### 4. 2D Neighbor Merging (비교군)

복잡한 유사도 계산 대신, 쿼리($Q$) 토큰들에 대해 $2 \times 2$ 평균 풀링(Average Pooling)을 적용하여 토큰 수를 75% 줄이는 단순한 전략이다. 이는 Segformer++의 스마트한 병합 전략이 실제로 어떤 이득을 주는지 분석하기 위한 대조군으로 활용된다.

### 5. 계산 복잡도 분석

Segformer++의 전체 계산 복잡도는 다음과 같이 표현된다.
$$O\left(\frac{N^2}{\lambda^{(kv)}\lambda^q R^2}D + \frac{N^2}{4}D + \frac{N^2}{4R^4}D\right) = O\left(\left(\frac{1}{\lambda^{(kv)}\lambda^q R^2} + \frac{0.25}{1 + \frac{R^4}{R^4}}\right)N^2 D\right)$$
결과적으로 일반적인 attention의 $O(N^2 D)$보다 훨씬 낮은 비용으로 연산이 가능하다.

## 📊 Results

### 1. 시맨틱 세그멘테이션 (Cityscapes, ADE20K)

- **Cityscapes (1024x1024)**: `Segformer++ HQ` 모델은 mIoU 성능을 거의 그대로 유지하면서 추론 속도를 **61% 향상(Speedup 1.61x)** 시켰다. `Segformer++ fast` 모델은 약간의 성능 저하가 있지만 속도를 **94% 향상(Speedup 1.94x)** 시켰다.
- **소형 객체 성능**: $\text{mIoU}_{\text{small}}$ 지표 분석 결과, 2D Neighbor Merging보다 Segformer++가 소형 객체 정보를 더 잘 보존함을 확인하였다.
- **학습 효율**: 학습 단계에서 적용했을 때, `Segformer++ fast`는 메모리 사용량을 48.3GB에서 30.5GB로 크게 줄였으며 학습 속도(Steps/s) 또한 향상되었다.

### 2. 인간 포즈 추정 (JBD, MS COCO)

- **JBD (640x480)**: `Segformer++ HQ`는 오리지널 Segformer와 거의 동일한 PCK 성능을 보이면서 추론 속도를 높였다. 특히 소형 디테일 표현에서 2D Neighbor Merging보다 우수한 결과를 보였다.

### 3. 해상도별 속도 향상 (Random Tensor 실험)

이미지 해상도가 높아질수록 Segformer++의 효율성이 극대화된다.

- $1024 \times 1024$ 해상도에서 `Segformer++ fast`는 **1.94배**의 속도 향상을 보였다.
- $3840 \times 2160$ (4K 수준) 고해상도에서는 **4.31배**까지 속도가 향상되어, 고해상도 데이터 처리에서 매우 강력한 이점을 가짐을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 Token Merging이 단순히 계산량을 줄이는 것을 넘어, 시맨틱 정보를 효과적으로 압축할 수 있음을 보여주었다. 특히 다음과 같은 통찰을 얻을 수 있다.

- **스마트 병합의 중요성**: 단순한 2D 풀링(Neighbor Merging)은 연산은 더 빠를 수 있으나, 소형 객체와 같은 세밀한 정보를 손실시킨다. 반면 유사도 기반의 병합은 중요한 특징을 유지하며 토큰을 줄인다.
- **구조적 제약의 해결**: Transformer의 효율성 전략을 Convolutional 기반 구조(Segformer)에 적용하기 위해서는 '병합 후 복원(Merge $\rightarrow$ Unmerge)' 과정이 필수적임을 확인하였다.
- **고해상도 적합성**: 입력 해상도가 증가할수록 $O(N^2)$의 부담이 커지므로, 토큰 수를 동적으로 줄이는 본 방법론은 실시간 고해상도 애플리케이션 및 엣지 디바이스 배포에 매우 유리하다.

한계점으로는, 최적의 감소율($r$)이 입력 이미지의 해상도에 따라 달라진다는 점이 언급되었다. 즉, 해상도가 바뀔 때마다 최적의 $r$ 값을 다시 설정해야 할 가능성이 있다.

## 📌 TL;DR

본 논문은 Segformer 아키텍처에 **Token Merging** 전략을 결합한 **Segformer++**를 제안한다. 유사한 토큰을 병합하고 Attention 후 다시 복원하는 방식을 통해, **재학습 없이도** 고해상도 이미지 추론 속도를 획기적으로(최대 2배 가까이) 높이면서 성능 저하를 최소화하였다. 특히 4K와 같은 초고해상도 이미지에서 효율성이 극대화되며, 시맨틱 세그멘테이션과 포즈 추정 등 다양한 dense pixel task에 즉시 적용 가능한 효율적인 드롭인(drop-in) 솔루션을 제공한다.
