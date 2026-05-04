# OneRing: A Simple Method for Source-free Open-partial Domain Adaptation

Shiqi Yang, Yaxing Wang, Kai Wang, Shangling Jui, Joost van de Weijer (2023)

## 🧩 Problem to Solve

본 논문은 Source-free Open-partial Domain Adaptation (SF-OPDA) 문제를 해결하고자 한다. 일반적인 Domain Adaptation (DA)은 소스 도메인의 레이블 데이터와 타겟 도메인의 레이블 없는 데이터를 동시에 사용하여 도메인 간의 분포 차이를 줄이는 것을 목표로 한다. 그러나 SF-OPDA 설정은 다음과 같은 세 가지 매우 까다로운 제약 조건을 동시에 가진다.

첫째, **Source-free** 설정으로 인해 데이터 프라이버시 보호 및 계산 자원 제한 등의 이유로 타겟 도메인 적응 과정에서 소스 데이터에 접근할 수 없다. 둘째, **Open-partial** 설정으로 인해 소스 도메인과 타겟 도메인 사이에 Category shift가 존재한다. 즉, 소스에만 존재하는 클래스, 타겟에만 존재하는 클래스(Unknown class), 그리고 두 도메인이 공유하는 클래스가 모두 섞여 있는 상황이다. 셋째, **Domain shift**로 인해 동일한 클래스라 하더라도 두 도메인 간의 데이터 분포가 서로 다르다.

따라서 본 연구의 목표는 소스 데이터 없이도 알려진 클래스(Known classes)와 알려지지 않은 클래스(Unknown classes)를 정확하게 구분하면서, 타겟 도메인의 분포에 성공적으로 적응하는 단순하고 효율적인 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 소스 학습 단계에서부터 **알려지지 않은 클래스를 예측할 수 있는 능력을 내재화한 $(n+1)$-way 분류기**를 설계하고, 타겟 적응 단계에서는 **가중치 기반의 Entropy Minimization**을 통해 소스 데이터 없이 도메인 적응을 수행하는 것이다.

중심적인 직관은 "어떤 알려진 클래스에 대해 가장 유사하지만 정답은 아닌 클래스가 곧 Unknown 클래스가 될 수 있다"는 가정이다. 이를 위해 분류기의 마지막 차원을 Unknown 전용 차원(OneRing dimension)으로 설정하고, 학습 시 정답 클래스 외의 가장 높은 점수가 이 Unknown 차원으로 향하도록 강제함으로써, 한 번도 본 적 없는 클래스를 거부(Reject)할 수 있는 능력을 갖추게 한다.

## 📎 Related Works

기존의 Open-set Recognition (OSR) 연구들은 주로 알려지지 않은 클래스를 탐지하기 위해 복잡한 밀도 추정 모듈을 도입하거나, 가상의 Unknown 샘플을 생성하여 학습시키는 방식을 사용하였다. 그러나 이러한 방식은 계산 비용이 많이 들며, 도메인 시프트가 발생하는 상황에서는 성능이 급격히 저하되는 한계가 있다.

또한, Universal Domain Adaptation (UniDA) 또는 Open-partial DA (OPDA) 분야의 기존 방법론들(예: OVANet)은 타겟 적응 과정에서 소스 데이터에 접근해야만 한다. 이는 실제 배포 환경에서 데이터 프라이버시 문제로 인해 적용하기 어렵다. 최근 제안된 Source-free DA (SFDA) 연구들은 주로 Closed-set 상황을 가정하거나, Unknown 클래스 탐지를 위해 별도의 샘플 합성 과정을 거쳐야 하는 복잡함이 있었다.

반면, OneRing은 추가적인 데이터 생성이나 복잡한 모듈 없이, 단순한 손실 함수 변경만으로 Open-set 능력을 확보하고 가중치 기반 엔트로피 최소화를 통해 Source-free 설정을 달성한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. Source Training: One Ring Classifier

소스 학습 단계에서는 $|C_s|$개의 알려진 클래스에 더해, Unknown 클래스를 위한 하나의 차원을 추가한 $(|C_s|+1)$-way 분류기를 학습시킨다. Unknown 클래스의 샘플이 없는 상태에서 이를 학습시키기 위해, 본 논문은 다음과 같은 수정된 Cross Entropy (CE) 손실 함수를 제안한다.

$$L_{source} = \mathbb{E}_{x_i \sim D_s} [L_{ce}(p(x_i), y_i) + L_{ce}(\hat{p}(x_i), \hat{y}_i)]$$

여기서 $p(x_i)$는 $(|C_s|+1)$ 차원의 예측 벡터이며, $y_i$는 실제 정답 클래스이다. $\hat{p}(x_i)$는 $p(x_i)$에서 정답 클래스에 해당하는 차원을 제외한 벡터이며, $\hat{y}_i$는 Unknown 클래스(마지막 차원)를 정답으로 하는 원-핫 벡터이다.

이 구조의 목적은 두 가지이다. 첫째, 가장 높은 로짓(logit)은 정답 클래스가 가져가게 한다. 둘째, 정답을 제외한 나머지 클래스들 중에서는 Unknown 클래스가 가장 높은 점수를 갖도록 강제한다. 이를 통해 모델은 학습 데이터에 없는 새로운 클래스가 들어왔을 때, 이를 Unknown 차원으로 분류할 수 있는 능력을 갖게 된다.

### 2. Target Adaptation: Weighted Entropy Minimization

소스 학습이 완료된 모델을 타겟 도메인에 적응시키기 위해, 본 논문은 소스 데이터 없이 타겟 데이터의 예측 불확실성을 줄이는 Entropy Minimization 방식을 사용한다. 다만, 알려진 클래스와 Unknown 클래스 간의 불균형을 해소하기 위해 다음과 같은 가중치 기반 손실 함수를 제안한다.

$$L_{target} = \frac{bs}{\hat{n}_{all}^k} \mathbb{E}_{\bar{y}_i \in C_s} L_{ent}(p(x_i)) + \frac{bs}{\hat{n}_{all}^u} \mathbb{E}_{\bar{y}_i \in C_u} L_{ent}(p(x_i))$$

- $bs$: 배치 크기 (batch size)
- $\hat{n}_{all}^k, \hat{n}_{all}^u$: 전체 데이터셋에서 각각 Known 클래스와 Unknown 클래스로 예측된 샘플의 수
- $L_{ent}$: 엔트로피 손실 함수

이 가중치 $\frac{bs}{\hat{n}_{all}^k}$와 $\frac{bs}{\hat{n}_{all}^u}$는 타겟 도메인 내에서 Known/Unknown 클래스의 비율을 고려하여 두 엔트로피 항의 균형을 맞추는 역할을 한다.

### 3. Augmentation with Attracting-and-Dispersing (AaD)

성능을 더욱 높이기 위해, 본 논문은 Closed-set SFDA의 SOTA 방법론인 AaD를 결합한 $\text{OneRing}^+$ 버전을 제안한다. AaD는 특징 공간에서 동일 클래스는 모으고(Attracting) 서로 다른 클래스는 멀어지게(Dispersing) 하는 $L_{dis}$와 $L_{div}$ 손실 함수를 사용한다.

$$L_{target+} = \frac{bs}{\hat{n}_{all}^k} \mathbb{E}_{\bar{y}_i \in C_s} [L_{ent}(p(x_i)) + L_{dis} + L_{div}] + \frac{bs}{\hat{n}_{all}^u} \mathbb{E}_{\bar{y}_i \in C_u} [L_{ent}(p(x_i)) + L_{dis}]$$

단, Unknown 클래스는 단일 클래스로 간주되므로 다양성 항인 $L_{div}$는 적용하지 않는다.

## 📊 Results

### 실험 설정

- **데이터셋**: Office-31, Office-Home, VisDA, DomainNet을 사용하여 SF-OPDA 설정에서 평가하였다.
- **지표**: 알려진 클래스의 정확도(OS)와 Unknown 클래스의 정확도(UNK)를 모두 고려한 Harmonic Mean (H-score)을 주요 지표로 사용하였다.
- **비교 대상**: OVANet, DCC, CMU 등 기존 OPDA 방법론들과 비교하였다.

### 주요 결과

1. **정량적 성능**: $\text{OneRing}$은 소스 데이터 없이도 기존의 소스 데이터를 사용하는 OPDA 방법론들을 대부분 능가하였다. 특히 AaD를 결합한 $\text{OneRing}^+$는 Office-31, Office-Home, VisDA 데이터셋에서 SOTA인 OVANet보다 각각 2.5%, 7.2%, 13% 더 높은 성능 향상을 보였다.
2. **Unknown 클래스 수에 대한 강건성**: 타겟 도메인의 Unknown 클래스 개수를 변화시키며 실험한 결과, OneRing은 클래스 수의 변화에도 성능 저하가 적고 매우 강건한 모습을 보였다.
3. **가중치의 중요성**: 엔트로피 최소화 식에서 가중치를 제거했을 때 성능이 크게 하락하는 것을 확인하여, Known/Unknown 비율을 맞추는 가중치 설계가 필수적임을 입증하였다.
4. **시각화 분석**: t-SNE 분석 결과, OneRing 분류기의 Unknown 프로토타입이 알려진 클래스들의 특징 군집으로부터 멀리 떨어져 위치하며, 실제 Unknown 샘플들이 이 프로토타입 근처로 모이는 것을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 복잡한 모듈이나 데이터 생성 없이, 손실 함수의 단순한 변경만으로 Source-free 환경에서 Open-partial DA를 수행할 수 있음을 보여주었다. 특히 $(n+1)$-way 분류기 설계는 OSR(Open-set Recognition) 분야의 핵심 난제인 "보지 못한 클래스의 거부" 문제를 매우 효율적으로 해결하였다.

**강점**:

- 소스 데이터가 필요 없으므로 프라이버시 문제가 해결된다.
- 구현이 매우 단순하며 추가적인 하이퍼파라미터 튜닝 부담이 적다.
- 기존의 Closed-set SFDA 기법(AaD 등)과 쉽게 결합하여 성능을 확장할 수 있다.

**한계 및 논의**:

- 소스 학습 시 두 개의 CE 손실 함수를 사용하는데, DomainNet과 같은 매우 큰 데이터셋에서는 수렴에 어려움이 있어 2단계 학습(Standard CE $\rightarrow$ OneRing CE)이 필요했다는 점이 언급된다. 이는 손실 함수 간의 충돌 가능성을 시사한다.
- 타겟 적응 단계에서 분류기 헤드를 고정(fixed)하고 특징 추출기만 학습시키는데, 분류기 헤드까지 함께 미세 조정했을 때의 효과에 대해서는 명시적으로 다루지 않았다.

## 📌 TL;DR

본 논문은 소스 데이터 없이 도메인 및 카테고리 시프트를 해결하는 **Source-free Open-partial Domain Adaptation (SF-OPDA)**를 위한 **OneRing** 방법론을 제안한다. 핵심은 소스 학습 시 Unknown 클래스를 위한 가상 차원을 만들어 정답 외의 가장 높은 점수를 배정하는 $(n+1)$-way 분류기를 학습시키고, 타겟 적응 시 가중치 기반 엔트로피 최소화를 사용하는 것이다. 이 단순한 접근법은 소스 데이터를 사용하는 기존 SOTA 모델인 OVANet보다 우수한 성능을 보였으며, 특히 SFDA 기법과 결합 시 강력한 성능을 발휘한다. 이는 향후 데이터 프라이버시가 중요한 실전 환경의 도메인 적응 연구에 중요한 기여를 할 것으로 보인다.
