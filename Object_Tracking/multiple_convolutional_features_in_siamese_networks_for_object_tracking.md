# Multiple Convolutional Features in Siamese Networks for Object Tracking

Zhenxi Li, Guillaume-Alexandre Bilodeau, Wassim Bouachir (2021)

## 🧩 Problem to Solve

본 논문은 Visual Object Tracking (VOT) 분야에서 Siamese 네트워크 기반 트래커들이 갖는 고유한 한계점을 해결하고자 한다. Siamese 트래커들은 일반적으로 연산 속도와 정확도 사이의 균형이 뛰어나 높은 성능을 보이지만, 대부분의 경우 CNN의 마지막 Convolutional layer에서 추출된 특징(feature)만을 사용하여 유사도 분석 및 타겟 탐색을 수행한다.

마지막 레이어의 특징은 높은 수준의 의미론적 정보(semantic information)를 담고 있어 일반화 능력이 좋지만, 수용 영역(receptive field)이 넓어 해상도가 낮고 세부적인 공간 정보가 손실된다는 단점이 있다. 이는 결과적으로 타겟의 정밀한 위치 추정(localization)을 어렵게 만들며, 타겟의 외형 변화(appearance variations)가 심한 까다로운 시나리오에서 추적 성능을 저하시키는 원인이 된다. 따라서 본 연구의 목표는 다양한 계층의 특징과 다중 모델을 결합하여 더욱 강건하고 정밀한 타겟 표현(representation)을 구축하는 Multiple Features-Siamese Tracker (MFST)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단일 레이어의 특징만을 사용하는 대신, CNN의 여러 계층에서 생성되는 계층적 특징(hierarchical features)과 서로 다른 두 가지 CNN 모델의 특징을 결합하여 타겟의 표현력을 극대화하는 것이다.

주요 기여 사항은 다음과 같다.

1. **계층적 특징 및 다중 모델 활용**: 서로 다른 추상화 수준을 가진 여러 Convolutional layer의 특징과, 서로 다른 태스크로 학습된 두 모델(SiamFC, AlexNet)의 특징을 동시에 활용하여 표현의 다양성을 확보하였다.
2. **특징 재교정 모듈(Feature Recalibration Module) 도입**: Squeeze-and-Excitation (SE) 블록을 통해 각 채널의 중요도를 학습하고 특징을 재교정함으로써, 추적 작업에 더 유용한 특징이 강조되도록 설계하였다.
3. **반응 맵 융합 전략(Response Map Fusion Strategies)**: 서로 다른 레이어와 모델에서 생성된 여러 개의 반응 맵(response map)을 효과적으로 결합하기 위한 하드 가중치(Hard Weight), 소프트 평균(Soft Mean), 소프트 가중치(Soft Weight) 전략을 제안하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 검토하고 차별점을 제시한다.

1. **Siamese Trackers**: SiamFC와 같은 선구적인 연구들은 두 개의 동일한 네트워크 브랜치를 통해 유사도 학습 문제로 추적을 정의하였다. 이후 CFNet, MBST, SA-Siam 등이 제안되었으나, 대부분 마지막 컨볼루션 레이어의 출력만을 사용한다는 공통적인 한계가 있다. 본 논문은 이를 극복하기 위해 초기 레이어의 세부 정보를 결합한다.
2. **Hierarchical Convolutional Features**: HCFT나 SiamRPN++와 같이 여러 레이어의 특징을 결합하는 시도가 있었으며, 이는 단순한 단일 레이어 특징보다 표현 능력이 뛰어남이 입증되었다. 본 논문은 이러한 계층적 구조를 Siamese 프레임워크에 통합하고 여기에 재교정 모듈을 추가하였다.
3. **Multi-Branch Tracking**: TRACA, MDNet, MBST 등은 타겟 외형 변화에 대응하기 위해 다중 브랜치나 전문가 네트워크를 사용하였다. 하지만 이는 계산 비용을 증가시키는 경향이 있다. MFST는 동일한 모델 내의 여러 레이어를 활용함으로써 추가적인 계산 비용을 최소화하면서도 특징의 다양성을 확보하는 전략을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조

MFST는 두 개의 사전 학습된 CNN 모델인 SiamFC($S$)와 AlexNet($A$)을 특징 추출기로 사용한다. 입력으로는 초기 프레임 또는 이전 프레임에서 얻은 타겟 템플릿(exemplar patch, $z$)과 현재 프레임의 탐색 영역(search region, $x$)이 주어진다.

### 특징 추출 및 재교정 (Feature Recalibration)

각 모델에서 $\text{conv3}, \text{conv4}, \text{conv5}$ 레이어로부터 기초 특징을 추출한다. 추출된 특징 $\text{F}$는 Squeeze-and-Excitation (SE) 블록을 통해 재교정된다. SE-블록은 다음 두 단계로 구성된다.

1. **Squeeze**: Global Average Pooling을 통해 $W \times H \times C$ 크기의 특징 맵을 $1 \times 1 \times C$ 크기의 채널 기술자(channel descriptor) $\omega^{sq}$로 압축한다.
   $$\omega^{sq} = \frac{1}{W \times H} \sum_{m=1}^{W} \sum_{n=1}^{H} v^c(m,n), \quad (c=1, \dots, C)$$
2. **Excitation**: 두 개의 MLP(Multi-Layer Perceptron) 레이어와 Sigmoid 활성화 함수를 통해 채널 간의 의존성을 학습하여 채널 가중치 $\omega^{ex}$를 생성한다.
   $$\omega^{ex} = \sigma(W_2 \delta(W_1 \omega^{sq}))$$
   여기서 $\delta$는 ReLU 활성화 함수이며, $W_1, W_2$는 학습 가능한 가중치이다. 최종적으로 재교정된 특징 $F^*$는 다음과 같이 계산된다.
   $$F^*_l = \omega^{ex} \cdot F_l$$

### 반응 맵 생성 및 결합 (Response Maps Combination)

재교정된 특징들을 사용하여 타겟 $z$와 탐색 영역 $x$ 사이의 cross-correlation을 수행하여 반응 맵 $r$을 생성한다.
$$r(z,x) = \text{corr}(F^*(z), F^*(x))$$
두 모델과 세 개의 레이어를 사용하므로 총 6개의 반응 맵($r^S_{c3}, r^S_{c4}, r^S_{c5}, r^A_{c3}, r^A_{c4}, r^A_{c5}$)이 생성된다. 이 맵들은 다음과 같은 세 가지 전략 중 하나로 융합된다.

- **Hard Weight (HW)**: 각 맵에 실험적으로 결정된 고정 가중치 $w_t$를 곱하여 합산한다.
  $$r^* = \sum_{t=1}^{N} w_t r_t$$
- **Soft Mean (SM)**: 각 맵을 자신의 최댓값으로 정규화하여 평균을 낸다.
  $$r^* = \sum_{t=1}^{N} \frac{r_t}{\max(r_t)}$$
- **Soft Weight (SW)**: 정규화된 맵에 가중치를 곱하여 합산한다.
  $$r^* = \sum_{t=1}^{N} w_t \frac{r_t}{\max(r_t)}$$

최종 융합된 반응 맵에서 최댓값을 가지는 위치가 타겟의 새로운 중심 좌표가 된다.

### 학습 절차

SiamFC 모델과 SE-블록은 ImageNet 비디오 데이터셋을 사용하여 Logistic Loss로 학습되었다.
$$L(y,v) = \frac{1}{|r|} \sum_{u \in r} \log(1 + \exp(-y v))$$
여기서 $y$는 ground-truth 라벨(양성 1, 음성 -1)이며, $v$는 반응 맵의 스코어이다. SE-블록 학습 시 기본 CNN 모델의 파라미터는 고정(freeze)시켰다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB-2013, OTB-50, OTB-100 및 VOT2018 벤치마크를 사용하였다.
- **평가 지표**: OTB에서는 Precision(중심 위치 오차)과 Success(IoU 기반 AUC)를 측정하였으며, VOT2018에서는 Accuracy(A), Robustness(R), EAO를 측정하였다.
- **구현 환경**: Nvidia Titan X GPU를 사용하였으며, 평균 처리 속도는 $39\text{ fps}$이다.

### 주요 결과

1. **Ablation Study**:
   - 단일 레이어 특징만 사용할 때보다 여러 레이어를 적절히 결합했을 때 표현력이 크게 향상되었다.
   - SE-블록을 통한 재교정이 특징의 변별력을 높여 성능 향상에 기여함을 확인하였다.
   - 단일 모델(SiamFC 또는 AlexNet만 사용)보다 두 모델을 모두 사용할 때 가장 높은 성능을 보였다.
   - 융합 전략 중에서는 일반적으로 Soft Weight(SW) 방식이 가장 효과적이었다.
2. **정량적 비교**:
   - OTB 벤치마크에서 MFST는 baseline인 SiamFC를 크게 상회하며, MBST, CFNet 등 최신 SOTA 트래커들과 경쟁력 있는 성능을 보였다.
   - VOT2018 결과, 특히 Robustness(R) 지표에서 SiamRPN++보다 더 우수한 성능을 보여, 타겟을 놓치는 횟수가 적고 강건한 추적이 가능함을 입증하였다.
   - 속도 측면에서는 두 개의 모델을 사용하므로 SiamFC보다는 느리지만, 다중 브랜치를 사용하는 MBST보다는 빠르면서 정확도는 더 높았다.

## 🧠 Insights & Discussion

본 논문은 Siamese 네트워크의 단순한 구조를 유지하면서도, 특징 추출 단계에서 계층적 구조와 다중 모델의 다양성을 도입함으로써 성능을 높일 수 있음을 보여주었다. 특히 SE-블록을 통해 채널별 중요도를 동적으로 조절한 점은 Siamese 트래커가 타겟의 외형 변화에 더 유연하게 대응하게 만든 핵심 요소로 판단된다.

**강점**:

- 복잡한 Region Proposal Network(RPN) 없이도 계층적 특징 융합만으로 높은 강건성을 확보하였다.
- 서로 다른 태스크(추적 vs 분류)로 학습된 모델을 결합하여 특징의 상보적 효과를 이끌어냈다.

**한계 및 논의**:

- **정확도(Accuracy)의 한계**: VOT2018 결과에서 SiamRPN++보다 Robustness는 높지만 Accuracy는 다소 낮게 나타났다. 이는 저자들도 언급했듯이, RPN과 같은 제안 기반 방식이 정밀한 Bounding Box 회귀에는 더 유리하기 때문으로 보인다. 향후 MFST 구조에 RPN을 통합한다면 정확도를 더욱 높일 수 있을 것이다.
- **가중치의 설정**: 반응 맵 융합 시 사용된 가중치 $w_t$가 실험적으로 결정된 empirical weight라는 점은, 데이터셋이나 환경 변화에 따라 최적의 가중치가 달라질 수 있음을 시사하며, 이를 자동화하여 학습하는 메커니즘이 필요할 수 있다.

## 📌 TL;DR

본 논문은 Siamese 트래커가 마지막 레이어의 특징만 사용하여 발생하는 정밀도 저하 및 외형 변화 취약성 문제를 해결하기 위해, **두 개의 서로 다른 CNN 모델(SiamFC, AlexNet)**과 **다양한 계층의 특징($c3, c4, c5$)**을 결합한 **MFST**를 제안한다. 특히 **SE-블록을 통한 특징 재교정**과 **최적화된 융합 전략**을 통해 OTB 및 VOT2018 벤치마크에서 기존 SiamFC 대비 비약적인 성능 향상을 이루었으며, 실시간성($39\text{ fps}$)과 강건함을 동시에 확보하였다. 이 연구는 향후 Siamese 기반 트래커들이 더 풍부한 특징 표현을 구축하는 방향으로 나아가는 데 중요한 기초를 제공한다.
