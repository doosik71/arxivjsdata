# Unifying Instance and Panoptic Segmentation with Dynamic Rank-1 Convolutions

Hao Chen, Chunhua Shen, Zhi Tian (2020)

## 🧩 Problem to Solve

본 논문은 Instance Segmentation과 Semantic Segmentation을 하나의 통합된 프레임워크로 처리하는 Panoptic Segmentation의 효율성과 성능을 동시에 개선하는 것을 목표로 한다. 기존의 Panoptic Segmentation 접근 방식은 주로 'stuff'(배경)를 위한 semantic segmentation 네트워크와 'thing'(개별 객체)을 위한 instance segmentation 네트워크라는 두 개의 독립적인 브랜치를 사용하는 구조에 의존한다. 이러한 방식은 다음과 같은 문제점을 가진다.

첫째, 두 네트워크를 별도로 운영함으로써 발생하는 표현의 중복성(representation redundancy)이 크며, 이는 계산 자원이 제한된 환경(자율주행, 드론 등)에서 실시간 적용을 어렵게 만든다. 둘째, 기존의 Fully-Convolutional Instance Segmentation 모델(예: BlendMask, CondInst)은 dynamic module의 파라미터 확장성 문제로 인해 feature channel 수를 매우 작게 유지해야 하며, 이는 풍부한 클래스 정보를 인코딩해야 하는 semantic segmentation 작업에 적용하기 어렵게 만든다.

따라서 본 연구의 목표는 높은 효율성을 유지하면서도 두 작업을 통합하여 처리할 수 있는 단일 브랜치 기반의 fully-convolutional panoptic segmentation 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Dynamic Rank-1 Convolution (DR1Conv)**라는 새로운 동적 모듈을 제안하여 instance segmentation과 semantic segmentation의 특징 맵을 효율적으로 통합한 것이다.

DR1Conv의 중심 아이디어는 저차원 행렬 분해(low-rank factorization)를 컨볼루션 연산에 적용하여, 파라미터 수와 연산량을 최소화하면서도 고수준의 컨텍스트 정보(high-level context)와 저수준의 세부 특징(low-level detailed features)을 효과적으로 융합하는 것이다. 이를 통해 단일 공유 특징 맵을 사용하면서도 두 가지 세그멘테이션 작업을 동시에 수행하는 **DR1Mask** 프레임워크를 구현하였다. 또한, 텐서 분해 기반의 **Factored Attention**을 통해 instance segmentation의 예측 정확도를 높이면서도 연산 효율성을 확보하였다.

## 📎 Related Works

기존의 Panoptic Segmentation 연구는 주로 Mask R-CNN과 같은 two-stage 방식이나, stuff와 thing을 각각 처리한 후 융합하는 separate-branch 방식에 집중하였다. Panoptic-DeepLab과 같은 bottom-up 방식이 존재하지만, 여전히 두 작업에 대해 별도의 디코더를 사용하며 복잡한 데이터셋(COCO 등)에서 성능 한계가 있었다.

또한, YOLACT, BlendMask, CondInst와 같은 fully-convolutional instance segmentation 모델들은 dynamic module을 통해 instance 정보를 융합하지만, 이들은 대개 예측의 최종 단계에서만 융합을 수행하며, dynamic module의 파라미터 효율성이 낮아 채널 수를 크게 확장하지 못하는 한계가 있었다. 본 논문은 DR1Conv를 통해 이러한 효율성 문제를 해결하고, 이를 네트워크의 중간 레이어까지 확장하여 semantic segmentation 성능까지 동시에 향상시켰다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Dynamic Rank-1 Convolution (DR1Conv)

DR1Conv는 입력 특징 맵 $X$에 대해 두 개의 동적 팩터(dynamic factors) $A$와 $B$를 사용하여 가중치를 동적으로 조절하는 연산이다. $1 \times 1$ 컨볼루션의 경우, 위치 $(h, w)$에서의 연산은 다음과 같이 정의된다.

$$y_{hw} = (W(x_{hw} \circ a_{hw})) \circ b_{hw}$$

여기서 $\circ$는 element-wise multiplication을 의미하며, $W$는 정적 가중치, $a_{hw}$와 $b_{hw}$는 각각 텐서 $A$와 $B$의 해당 위치 값이다. 이를 임의의 커널 크기 $J \times K$로 일반화하면 전체 특징 맵 $Y$는 다음과 같이 계산된다.

$$Y = \text{DR1Conv}_{A, B}(X) = \text{Conv}(X \circ A) \circ B$$

이 구조는 정적 컨볼루션 전후로 동적 텐서를 곱해줌으로써, 매우 적은 연산량으로도 강력한 표현력을 가지며 특히 위치 민감도(position sensitivity)를 유지할 수 있다.

### 2. DR1Mask 전체 구조

DR1Mask는 크게 두 가지 브랜치로 구성된다.

- **Top-down Branch**: FCOS를 기반으로 하며, 각 인스턴스의 바운딩 박스 $b^{(i)}$, 인스턴스 임베딩 $e^{(i)}$, 그리고 multi-scale conditional feature pyramid $\{C^l = [A^l, B^l]\}$를 생성한다.
- **Bottom-up Branch (DR1Basis)**: DR1Conv를 기본 블록으로 사용하는 역피라미드 구조이다. FPN 특징 $P^l$과 컨텍스트 특징 $C^l$을 다음과 같이 융합하여 최종 세그멘테이션 특징 $F$를 생성한다.
$$F^l = \text{DR1Conv}_{A^l, B^l}(\text{Conv}_{3 \times 3}(P^l) + \uparrow_2(F^{l+1}))$$

### 3. Instance Prediction 및 Factored Attention

인스턴스 마스크 생성을 위해 RoIAlign으로 추출된 특징 $R^{(i)}$에 대해 **Factored Attention**을 적용한다. 기존 BlendMask의 full attention tensor $Q$의 중복성을 줄이기 위해, $Q$를 다음과 같이 저차원 분해한다.

$$Q^{(i)}_k = U^k \Sigma^{(i)}_k V^k$$

여기서 $U^k, V^k$는 모든 인스턴스가 공유하는 파라미터이며, $\Sigma^{(i)}_k$는 인스턴스별 임베딩 $s^{(i)}$에 의해 결정되는 대각 행렬이다. 이를 통해 임베딩 파라미터 수를 784개에서 16개로 획기적으로 줄이면서도 성능을 유지하였다.

### 4. Unified Panoptic Segmentation

Panoptic 결과를 얻기 위해 $1 \times 1$ 컨볼루션 레이어 $f_{pano}$를 추가한다. 이 레이어의 가중치 $W_{pano}$는 정적 가중치 $W_{stuff}$와 동적 가중치 $W_{thing}$으로 나뉜다.

$$W_{pano} = [W_{stuff}, W_{thing}]$$

$W_{stuff}$는 배경 클래스를 위해 고정된 값이며, $W_{thing}$은 예측된 인스턴스 임베딩들의 평균값 $\bar{e}_c$를 사용하여 동적으로 생성된다. 최종 출력 $Y_{pano}$는 다음과 같이 계산된다.

$$Y_{pano} = W_{pano}^T F$$

## 📊 Results

### 실험 설정

- **데이터셋**: MS COCO 2017 (80 thing, 53 stuff 클래스)
- **백본**: ResNet-50 및 ResNet-101
- **비교 대상**: Mask R-CNN, BlendMask, CondInst (Instance), Panoptic-DeepLab, UPSNet, SOGNet (Panoptic)
- **지표**: $\text{AP}$ (Instance), $\text{PQ}$ (Panoptic), $\text{SQ}, \text{RQ}$

### 주요 결과

1. **Instance Segmentation**: ResNet-101 백본과 Deformable Convolution을 사용했을 때, $\text{AP} 41.2\%$를 기록하며 기존 SOTA 모델인 CondInst 및 BlendMask보다 우수하거나 대등한 성능을 보였다. 특히 BlendMask 대비 약 10% 더 빠르고 1%p 높은 AP를 달성하였다.
2. **Panoptic Segmentation**: ResNet-50 기반 모델에서 $\text{PQ} 42.9$를 달성하여 Panoptic-DeepLab보다 8포인트 높은 성능을 보였으며, 실행 속도는 UPSNet 등 기존 2-브랜치 방식보다 약 2배 더 빨랐다.
3. **효율성**: DR1Mask는 단일 브랜치를 사용하므로, 'stuff' 세그멘테이션을 위한 추가 비용이 거의 없으며(단일 레이어 추가), 추론 시간 측면에서 매우 강력한 이점을 가진다.

## 🧠 Insights & Discussion

본 논문은 DR1Conv를 통해 고수준 컨텍스트와 저수준 특징의 융합이 semantic과 instance 작업 모두에 유익함을 입증하였다. 특히, 인스턴스 임베딩과 컨텍스트 정보 사이의 일관성을 유지하는 것이 성능 향상의 핵심임을 ablation study를 통해 확인하였다.

흥미로운 지점은 **Position Sensitive Attention**의 영향이다. 이 기법은 instance segmentation 성능은 높이지만, panoptic segmentation에서는 오히려 semantic segmentation 품질($\text{PQ}_{St}$)을 떨어뜨리는 결과를 초래하였다. 이는 stuff 영역까지 위치 민감한 인코딩을 강제하는 것이 불필요하고 오해의 소지가 있기 때문으로 분석된다.

또한, 네트워크의 stride로 인한 border padding 문제가 semantic segmentation의 경계 정확도를 떨어뜨린다는 점을 발견하고, 이를 위해 입력 사이즈의 가분성(divisibility)을 32에서 4로 낮추고 정렬된 업샘플링(aligned upsampling) 전략을 도입하여 해결하였다.

## 📌 TL;DR

본 논문은 **Dynamic Rank-1 Convolution (DR1Conv)**를 제안하여 Instance와 Semantic Segmentation을 하나의 효율적인 단일 브랜치로 통합한 **DR1Mask** 프레임워크를 제시한다. 이 모델은 저차원 행렬 분해를 통해 연산 효율성을 극대화하였으며, COCO 데이터셋에서 SOTA 수준의 성능과 기존 방식 대비 약 2배 빠른 추론 속도를 달성하였다. 이 연구는 복잡한 Panoptic Segmentation 작업을 단순하고 효율적인 구조로 통합할 수 있음을 보여주며, 향후 실시간 자율주행 및 로봇 제어 시스템의 세그멘테이션 모듈 설계에 중요한 참고 자료가 될 것으로 보인다.
