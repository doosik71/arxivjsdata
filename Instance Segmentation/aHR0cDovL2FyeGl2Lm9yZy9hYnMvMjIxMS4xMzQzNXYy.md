# A Benchmark of Long-tailed Instance Segmentation with Noisy Labels

Guanlin Li, Guowen Xu, Tianwei Zhang (2023)

## 🧩 Problem to Solve

본 논문은 **Label Noise(라벨 노이즈)**가 포함된 **Long-tailed(롱테일)** 데이터셋에서의 **Instance Segmentation** 문제를 다룬다. 일반적인 딥러닝 모델은 균형 잡힌 깨끗한 데이터셋을 가정하고 학습되지만, 실제 환경에서는 다음과 같은 두 가지 현실적인 문제에 직면한다.

첫째, 현실 세계에서 수집된 데이터는 대부분 소수의 클래스가 다수를 차지하고 대다수의 클래스는 적은 수의 샘플만을 가지는 Long-tailed 분포를 따른다. 이러한 불균형은 모델이 빈도수가 높은 'Head' 클래스에 편향되게 만들어, 'Body' 및 'Tail' 클래스에 대한 일반화 성능을 저하시킨다.

둘째, Instance Segmentation 데이터셋은 한 이미지 내에 많은 인스턴스가 존재하고 일부는 크기가 매우 작기 때문에, 주석 작업 과정에서 잘못된 라벨이 부여되는 Label Noise가 발생하기 쉽다. 특히 저비용의 전문성 낮은 작업자가 주석을 달 경우 이러한 경향이 심화된다.

따라서 본 논문의 목표는 Long-tailed 분포와 Label Noise가 동시에 존재하는 실제적인 상황을 모사한 벤치마크 데이터셋을 구축하고, 기존의 Long-tailed Instance Segmentation 알고리즘들이 이러한 노이즈 환경에서 얼마나 강건하게(robust) 동작하는지 체계적으로 평가하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Noisy-LVIS 데이터셋 구축**: 대규모 어휘(Large Vocabulary)를 가진 Long-tailed 데이터셋인 LVIS v1을 기반으로, 시맨틱 관계를 고려한 체계적인 라벨 노이즈를 추가하여 새로운 벤치마크 데이터셋을 제안한다.
2. **체계적인 성능 평가**: EQL, DropLoss, Seesaw Loss 등 기존의 대표적인 Long-tailed Instance Segmentation 손실 함수들을 다양한 노이즈 설정(Symmetric, Asymmetric) 및 샘플링 전략 하에서 평가하여 그 한계를 분석한다.
3. **실무적 통찰 제공**: 데이터 샘플링 방법(RFS 등)이 라벨 노이즈로 인한 분포 변화(Distribution Shift)로 인해 오히려 성능 저하를 일으킬 수 있음을 밝혀, 향후 노이즈에 강건한 롱테일 학습 연구의 필요성을 제시한다.

## 📎 Related Works

### LVIS v1

LVIS v1은 MS COCO 데이터셋을 기반으로 더 세밀한 주석을 추가한 데이터셋으로, 1,203개의 클래스를 포함하는 대표적인 Long-tailed Instance Segmentation 데이터셋이다.

### Long-tailed Instance Segmentation

기존 연구들은 Long-tailed 문제를 해결하기 위해 Gradient Calibration(예: EQL), Two-stage training, 데이터 증강, 새로운 모델 구조 등을 제안해 왔다. 하지만 이러한 방법들은 대부분 데이터셋이 깨끗하다는 가정하에 설계되었다.

### Noisy Label Learning

분류(Classification) 작업에서는 노이즈 라벨을 처리하기 위해 Noise Adaptation Layer, Robust Loss(예: MAE, GCE), Sample Selection(예: Co-teaching) 등의 기법이 연구되었다. 그러나 Instance Segmentation에서는 배경(Background)과 전경(Foreground)을 분리해서 처리해야 하는 특성이 있어, 분류 작업의 방법론을 그대로 적용하기 어렵다.

## 🛠️ Methodology

### 1. Noisy-LVIS 데이터셋 생성 과정

연구진은 LVIS v1의 1,203개 카테고리에 체계적으로 노이즈를 주입하기 위해 다음과 같은 파이프라인을 구축하였다.

**가. Super Class 정의 (Parse synset Categories)**
단순 랜덤 노이즈는 비현실적이므로, WordNet을 활용한 `nltk` 툴킷을 사용하여 카테고리 간의 상위어(Hypernym) 관계를 분석하였다. 이를 통해 전체 카테고리를 25개의 Super Class(예: food, living thing, commodity 등)와 하나의 'others' 클래스로 분류하였다.

**나. 노이즈 생성 알고리즘 (Noise Label Generation)**
두 가지 형태의 노이즈를 정의하여 주입한다.

- **Symmetric Noise (Class-agnostic)**: 원래 라벨과 상관없이 1,203개의 모든 클래스 중 하나를 무작위로 선택하여 교체한다.
- **Asymmetric Noise (Class-related)**: 원래 라벨이 속한 동일한 Super Class 내의 다른 카테고리를 무작위로 선택하여 교체한다. 이는 시맨틱적으로 유사한 클래스 간의 혼동을 모사하여 더 현실적인 노이즈를 생성한다.

노이즈 생성 절차는 다음과 같다.
$$\text{for each instance } i:$$
$$\text{if } r \sim U(0,1) \le p \text{ (noise ratio):}$$
$$\text{replace label } A[i] \text{ with a random class from the same super class } S[s]$$

### 2. 실험 설정 및 평가 방법

- **Backbone & Architecture**: ResNet-50 + FPN을 기본으로 하며, Mask R-CNN과 Cascade R-CNN 두 가지 구조를 사용한다.
- **평가 대상 방법론**:
  - **EQL / EQLv2**: Gradient balance를 통해 롱테일 문제를 해결하는 방법.
  - **DropLoss**: 특정 조건에서 손실을 제외하여 학습하는 방법.
  - **Seesaw Loss**: 클래스 간의 균형을 맞추는 가중치를 동적으로 조정하는 방법.
  - (일부 방법론은 Gumbel Optimized Loss를 적용하여 Sigmoid를 Gumbel 활성화 함수로 대체함)
- **샘플러 (Sampler)**:
  - **Random Sampler**: 무작위 추출.
  - **RFS (Repeat Factor Sampling)**: 빈도가 낮은 클래스를 더 많이 샘플링하는 전략.
- **평가 지표**: $AP$ (Average Precision)를 전체, Rare ($AP_r$), Common ($AP_c$), Frequent ($AP_f$) 클래스로 나누어 측정한다.

## 📊 Results

### 1. 라벨 노이즈가 분포에 미치는 영향

노이즈가 추가되면 원래의 Long-tailed 분포가 변하게 된다. 특히 RFS 샘플러는 빈도가 낮은 클래스를 오버샘플링하는데, 노이즈로 인해 '희귀 클래스'로 잘못 분류된 샘플들이 더 많이 샘플링되는 **Distribution Shift** 현상이 발생한다.

### 2. 샘플러 및 손실 함수의 영향 (Seesaw Loss 기준)

- **Clean 데이터**: RFS 샘플러가 Random 샘플러보다 성능을 크게 향상시킨다.
- **Noisy 데이터**: 노이즈 비율이 높아질수록 RFS의 이점이 약화된다. 이는 잘못된 라벨을 가진 샘플이 과도하게 학습에 참여하기 때문이다.

### 3. 방법론별 강건성 비교

- **전반적 경향**: 기존의 Long-tailed 전용 손실 함수들은 라벨 노이즈에 취약하며, 특히 희귀 클래스($AP_r$)의 성능이 급격히 하락한다.
- **노이즈 유형별 특성**:
  - **Symmetric Noise**: Seesaw Loss가 상대적으로 더 강건한 모습을 보인다.
  - **Asymmetric Noise**: DropLoss + Gumbel 조합이 상대적으로 더 강건하다.
- **결론**: 모든 유형의 노이즈에 대해 일관되게 우수한 성능을 내는 단일 손실 함수는 존재하지 않았으며, 이는 노이즈가 포함된 롱테일 데이터셋을 위한 새로운 손실 함수 설계가 매우 도전적인 과제임을 시사한다.

## 🧠 Insights & Discussion

본 논문은 Long-tailed 학습과 Noisy Label 학습이라는 두 가지 난제가 결합되었을 때 발생하는 시너지 효과(부정적인 방향으로)를 실험적으로 증명하였다.

**강점 및 발견**:

- 단순히 데이터를 섞는 것이 아니라 WordNet의 계층 구조를 이용해 Asymmetric Noise를 생성함으로써, 실제 주석 과정에서 발생할 수 있는 '유사 클래스 간 혼동'이라는 현실적인 문제를 벤치마크에 반영하였다.
- 롱테일 문제 해결을 위해 필수적으로 사용되던 RFS(Repeat Factor Sampling)가 노이즈 환경에서는 오히려 독이 될 수 있다는 점을 정량적으로 보여주었다.

**한계 및 논의**:

- 본 논문은 데이터셋 구축과 기존 방법론의 평가에 집중하고 있으며, 이를 완전히 해결할 새로운 알고리즘을 제안하지는 않았다.
- False Negative(전경 객체가 배경으로 처리됨) 노이즈는 고려하지 않고, 오직 잘못된 클래스로 지정된 노이즈에만 집중하였다. 실제 데이터셋에서는 두 가지 노이즈가 동시에 존재할 가능성이 높다.

## 📌 TL;DR

본 연구는 LVIS v1 데이터셋에 시맨틱 관계를 고려한 노이즈를 주입하여 **Long-tailed & Noisy Instance Segmentation 벤치마크**를 제안하였다. 실험 결과, 기존의 롱테일 해결 기법(EQL, Seesaw Loss 등)들은 라벨 노이즈에 매우 취약하며, 특히 빈도 기반 샘플링 전략이 노이즈로 인한 분포 왜곡을 심화시켜 성능을 저하시킨다는 것을 발견하였다. 이 연구는 실제 환경의 불완전한 데이터셋에서도 작동하는 강건한 인스턴스 분할 모델 연구의 필요성을 강조한다.
