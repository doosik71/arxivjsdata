# MFST: Multi-Features Siamese Tracker

Zhenxi Li, Guillaume-Alexandre Bilodeau, Wassim Bouachir (2021)

## 🧩 Problem to Solve

본 논문은 기존 Siamese Tracker들이 가진 특징 표현(Feature Representation)의 한계를 해결하고자 한다. 대부분의 Siamese Tracker들은 이미지 유사도 분석과 타겟 검색을 위해 CNN의 마지막 컨볼루션 층(last convolutional layer)에서 추출된 특징만을 사용한다.

하지만 마지막 층의 특징은 추상화 수준이 매우 높아 일반적인 특성은 잘 포착하지만, 해상도가 낮아 정밀한 위치 추적(precise localization)에는 불리하다는 단점이 있다. 반면, 앞쪽 층의 특징들은 저수준(low-level)의 세밀한 공간적 세부 정보를 포함하고 있어 정밀한 위치 추정에 매우 유용하다. 따라서 단일 계층의 특징만을 사용하는 것은 딥 유사도 학습(deep similarity learning) 프레임워크 내에서 최적의 선택이 아니며, 이는 특히 타겟의 외형 변화(appearance variation)가 심한 상황에서 성능 저하를 야기한다. 본 논문의 목표는 여러 계층의 계층적 특징(hierarchical features)과 서로 다른 모델의 특징을 융합하여 더욱 강건하고 정밀한 추적 알고리즘인 MFST를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 단일 모델의 단일 계층 특징에 의존하는 대신, **다양한 계층의 특징(Hierarchical Features)**과 **서로 다른 두 가지 CNN 모델(Diverse Models)**을 결합하여 풍부한 표현력을 확보하는 것이다.

중심적인 설계 아이디어는 다음과 같다:

1. **계층적 특징 활용**: CNN의 `conv3`, `conv4`, `conv5` 층에서 특징을 모두 추출하여 저수준의 공간 정보와 고수준의 의미 정보를 동시에 활용한다.
2. **다중 모델 앙상블**: 추적 목적의 `SiamFC`와 분류 목적의 `AlexNet`이라는 서로 다른 성격의 모델을 사용하여 외형 변화에 대한 강건성을 높인다.
3. **특징 재보정(Feature Recalibration)**: Squeeze-and-Excitation(SE) 블록을 도입하여 각 채널의 중요도를 학습하고 특징 맵을 재보정함으로써 표현력을 극대화한다.
4. **계층적 응답 맵 융합**: 각 층과 모델에서 생성된 여러 개의 응답 맵(response maps)을 최적의 전략(Hard Weight, Soft Mean, Soft Weight)으로 융합하여 최종 타겟 위치를 결정한다.

## 📎 Related Works

### 1. Deep Similarity Tracking

SiamFC와 같은 Siamese Tracker들은 오프라인 단계에서 일반적인 유사도 함수를 학습하고, 온라인 단계에서 템플릿과 검색 영역 간의 상호 상관(cross-correlation)을 통해 타겟을 찾는다. CFNet, SA-Siam 등이 제안되었으나, 이들은 여전히 마지막 컨볼루션 층의 출력에만 의존한다는 한계가 있다.

### 2. Exploiting Multiple Hierarchical Levels

HCFT 등의 연구는 CNN의 서로 다른 층이 서로 다른 수준의 시각적 추상화를 포함하고 있음을 보였으며, 여러 계층의 특징을 결합하는 것이 추적의 강건성을 높인다는 점을 입증하였다.

### 3. Multi-Branch Tracking

TRACA, MDNet, MBST 등은 타겟의 외형 변화를 해결하기 위해 여러 개의 브랜치(분기)를 사용한다. 특히 MBST는 여러 CNN 모델에서 특징을 추출하여 가장 판별력이 좋은 브랜치를 선택한다. 그러나 브랜치 수가 많아질수록 계산 비용이 증가하는 문제가 있다. MFST는 모델 수를 적게 유지하면서도 계층적 특징을 활용함으로써 낮은 계산 비용으로 유사한 효과를 거두고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

MFST의 구조는 입력 이미지(템플릿 $z$, 검색 영역 $x$)를 두 개의 사전 학습된 CNN 모델(`SiamFC` 및 `AlexNet`)에 통과시켜 특징을 추출하고, 이를 SE-블록으로 재보정한 후, 상호 상관 연산을 통해 응답 맵을 생성하고 최종적으로 이를 융합하는 순서로 구성된다.

### 주요 구성 요소 및 절차

#### 1. 특징 추출 및 재보정 (Feature Extraction & Recalibration)

두 모델 $\text{S}$(`SiamFC`)와 $\text{A}$(`AlexNet`)에서 각각 `conv3`, `conv4`, `conv5` 층의 특징 $\text{S}_l^i, \text{A}_l^i$를 추출한다. 각 특징 맵은 SE-블록을 통해 재보정된다.

**Squeeze 단계**: Global Average Pooling을 통해 채널 기술자 $\omega_{sq}$를 생성한다.
$$\omega_{sq} = \frac{1}{W \times H} \sum_{m=1}^{W} \sum_{n=1}^{H} v_c(m,n), (c=1, \dots, C)$$

**Excitation 단계**: 두 개의 MLP 층을 통해 채널 간의 의존성을 캡처하여 가중치 $\omega_{ex}$를 생성한다.
$$\omega_{ex} = \sigma(W_2 \delta(W_1 \omega_{sq}))$$
여기서 $\sigma$는 Sigmoid, $\delta$는 ReLU 활성화 함수이다.

**최종 재보정 특징**:
$$F_l^{i*} = \omega_{ex} \cdot F_l^i$$

#### 2. 응답 맵 생성 (Response Map Generation)

재보정된 특징 맵을 사용하여 템플릿과 검색 영역 간의 상호 상관 연산을 수행한다.
$$r(z, x) = \text{corr}(F^*(z), F^*(x))$$
이 과정을 통해 총 6개(2개 모델 $\times$ 3개 층)의 응답 맵이 생성된다.

#### 3. 응답 맵 융합 (Combining Response Maps)

생성된 응답 맵들을 다음 세 가지 전략 중 최적인 것을 선택하여 융합한다.

- **Hard Weight (HW)**: 각 맵에 고정된 가중치 $w_t$를 곱해 합산한다.
  $$r^* = \sum_{t=1}^{N} w_t r_t$$
- **Soft Mean (SM)**: 각 맵을 자신의 최댓값으로 정규화하여 평균을 낸다.
  $$r^* = \sum_{t=1}^{N} \frac{r_t}{\max(r_t)}$$
- **Soft Weight (SW)**: 정규화된 맵에 가중치를 적용하여 합산한다.
  $$r^* = \sum_{t=1}^{N} w_t \frac{r_t}{\max(r_t)}$$

최종적으로 융합된 응답 맵 $r^*$에서 최댓값을 갖는 위치가 새로운 타겟의 위치가 된다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB50, OTB2013, OTB100 벤치마크 사용.
- **평가 지표**: Center Location Error (CLE) 기준의 Precision plot과 IoU 기준의 Success plot (AUC) 사용.
- **구현 세부사항**: Nvidia Titan Xp GPU 사용, 평균 추적 속도는 39 fps.

### 주요 결과

1. **Ablation Study**:
   - 단일 층 특징보다 여러 층의 특징을 결합했을 때 성능이 크게 향상됨을 확인하였다.
   - SE-블록을 통한 재보정이 특징 표현력을 높여 성능 향상에 기여하였다.
   - 단일 모델을 사용할 때보다 `SiamFC`와 `AlexNet` 두 모델을 결합했을 때 더 높은 판별력을 보였다.
   - 융합 전략 중에서는 일반적으로 Soft Weight (SW) 전략이 가장 우수한 성능을 보였다.

2. **SOTA 비교**:
   - MFST는 OTB 벤치마크에서 MBST, LMCF, CFNet, SiamFC 등 기존 최신 추적기들보다 우수한 성능을 기록하였다 (OTB-50의 Precision 제외).
   - 특히 MBST가 많은 수의 CNN 모델 브랜치를 사용함에도 불구하고, MFST는 단 2개의 모델만으로 더 높은 정확도와 속도를 달성하였다.

## 🧠 Insights & Discussion

본 논문은 Siamese Tracker의 고질적인 문제인 '단일 계층 특징 사용'이 정밀도와 강건성을 제한한다는 점을 정확히 짚어내었다.

**강점**:

- 저수준 특징(공간 세부 정보)과 고수준 특징(의미 정보)을 계층적으로 융합함으로써, 타겟의 외형이 변하거나 정밀한 위치 추적이 필요한 상황에서 매우 효율적인 대응이 가능하다.
- SE-블록을 통해 모델의 파라미터를 고정한 채로 채널 가중치만을 학습시켜 효율적으로 특징을 최적화하였다.
- 서로 다른 목적(추적 vs 분류)으로 학습된 모델을 결합하여 특징의 다양성을 확보한 점이 인상적이다.

**한계 및 논의**:

- 융합 전략(HW, SM, SW)과 가중치 $w_t$가 실험적인 경험치(empirical weights)에 기반하여 설정되었다는 점은 하이퍼파라미터 튜닝에 대한 의존도가 높음을 시사한다.
- 39 fps라는 속도는 실시간 추적은 가능하지만, 단순한 SiamFC보다는 느리며, 모델 수가 늘어남에 따라 계산 비용이 증가하는 트레이드-오프 관계가 존재한다.

## 📌 TL;DR

MFST는 기존 Siamese Tracker들이 마지막 컨볼루션 층만 사용하던 한계를 극복하기 위해, **두 가지 서로 다른 CNN 모델(`SiamFC`, `AlexNet`)의 여러 계층(`conv3~5`) 특징을 추출하고 SE-블록으로 재보정하여 융합**하는 방식의 추적기이다. 이를 통해 계산 효율성을 유지하면서도 타겟의 외형 변화에 강건하고 정밀한 추적 성능을 달성하였으며, OTB 벤치마크에서 기존 SOTA 모델들을 뛰어넘는 성능을 보였다. 이 연구는 향후 다중 스케일 특징 융합 및 모델 앙상블 기반의 실시간 추적 연구에 중요한 기초 자료가 될 가능성이 높다.
