# Improving Generalized Zero-Shot Learning by Semantic Discriminator

Xinpeng Li (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Generalized Zero-Shot Learning (GZSL) 환경에서 발생하는 **Seen class와 Unseen class 간의 도메인 불균형 및 오분류 문제**이다.

전통적인 Zero-Shot Learning (ZSL)은 테스트 단계에서 오직 Unseen class의 인스턴스만을 분류하지만, GZSL은 Seen class와 Unseen class가 모두 포함된 테스트 세트에서 동시에 분류를 수행해야 한다. 이 과정에서 모델은 Unseen class의 인스턴스를 Seen class로 잘못 분류하는 경향이 강하며, 이로 인해 Unseen class의 분류 정확도가 Seen class에 비해 현저히 낮게 나타나는 문제가 발생한다.

따라서 본 연구의 목표는 입력 인스턴스가 Seen 도메인에 속하는지 또는 Unseen 도메인에 속하는지를 정확하게 판별할 수 있는 **Semantic Discriminator (SD)**를 설계하여 GZSL의 전반적인 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스를 Semantic space로 투영했을 때, 해당 인스턴스가 속한 도메인(Seen vs Unseen)에 따라 **투영된 벡터의 노름(Norm) 길이와 Semantic embedding 벡터와의 거리**가 서로 다른 특성을 보인다는 관찰에서 시작된다.

구체적인 직관은 다음과 같다.

1. **Norm 길이의 차이**: Seen class의 인스턴스는 투영 후의 노름 길이가 Semantic embedding 벡터의 노름 길이와 유사하게 유지되지만, Unseen class의 인스턴스는 그 차이가 상대적으로 크게 나타난다.
2. **최소 거리의 차이**: Seen class의 인스턴스는 투영된 벡터와 Seen class의 Semantic embedding 벡터들 사이의 최소 거리(Minimum Semantic Distance, MSD)가 매우 작지만, Unseen class의 인스턴스는 이 거리가 상대적으로 멀다.

이러한 관찰을 바탕으로, 복잡한 파라미터 설정 없이 통계적 수치(평균, 표준편차)를 이용해 도메인을 판별하는 간단하고 효율적인 Semantic Discriminator를 제안하였다.

## 📎 Related Works

ZSL 및 GZSL은 기본적으로 시각적 특징과 시맨틱 정보(Attribute, Word vector 등)를 연결하는 Semantic embedding을 브릿지로 사용한다. 기존의 GZSL 접근 방식은 크게 두 가지로 나뉜다.

1. **통합 모델 방식**: Seen과 Unseen 클래스 모두에서 잘 작동하는 단일 모델을 구축하려 하며, 주로 GAN이나 VAE 같은 생성 모델을 사용하여 Unseen 클래스의 가상 특징을 생성하거나 직접적인 매핑 함수를 학습시킨다. 그러나 이 방식은 종종 Seen 클래스의 정확도를 희생시키는 경향이 있다.
2. **도메인 판별 방식**: 입력 인스턴스가 어느 도메인에 속하는지 먼저 판단한 뒤, 각각 다른 분류기(Fully supervised model vs ZSL model)를 적용하는 방식이다. 기존 연구들(Hard gating, Calibration, Temperature scaling 등)은 예측 점수의 분포를 이용해 도메인을 판별하려 했으나, Unseen 클래스의 점수 분포가 매우 분산되어 있어 판별 정확도가 낮고 구현이 복잡하다는 한계가 있다.

본 논문은 이러한 기존의 점수 기반 방식 대신, Semantic space에서의 기하학적 특성(노름 및 거리)을 이용함으로써 더 단순하고 일반화 성능이 높은 판별 방식을 제안하며 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

전체 시스템은 **Semantic Mapping (SM) $\rightarrow$ Semantic Discriminator (SD) $\rightarrow$ 분류 모듈 (FSC 또는 ZSL)** 순으로 구성된다.

1. **Semantic Mapping (SM)**: ResNet101을 통해 추출된 시각적 특징을 MLP를 통해 Semantic space로 투영한다.
2. **Semantic Discriminator (SD)**: 투영된 벡터를 분석하여 Seen 도메인인지 Unseen 도메인인지 결정한다.
3. **분류 단계**:
   - SD가 'Seen'으로 판별한 경우 $\rightarrow$ Fully Supervised Classification (FSC) 모델로 전달.
   - SD가 'Unseen'으로 판별한 경우 $\rightarrow$ ZSL 모델로 전달.

### 상세 구성 요소

#### 1. Semantic Mapping Module

시각적 특징 $g_c(x)$를 시맨틱 벡터 $z$로 매핑하는 함수 $F$를 학습시킨다. 손실 함수는 다음과 같은 $L_2$ norm 기반의 MSE(Mean Squared Error)를 사용한다.
$$L_{SM} = \frac{1}{N_s} \sum_{i=1}^{N_s} \|F(g_c(x_{s_i})) - z_{s_i}\|_2^2$$
여기서 $x_{s_i}$는 Seen class의 인스턴스, $z_{s_i}$는 해당 클래스의 시맨틱 임베딩 벡터이다.

#### 2. Semantic Discriminator (SD)

본 논문은 세 가지 도메인 판별 전략을 제안한다. 모든 전략에서 사용되는 기본 변수는 투영된 벡터와 통합 노름 $l$ 사이의 절대 차이 $D_l(x)$이다.
$$D_l(x) = |\|F(g_c(x))\|_2 - l|$$

- **전략 1: $\text{SD}_{OL}$ (Only by Length)**
  노름 차이 $D_l(x)$가 임계값 $R_{OL}$보다 작으면 Seen, 그렇지 않으면 Unseen으로 판별한다. 임계값은 Seen class의 통계값으로 결정한다.
  $$R_{OL} = m_{D_l} + var_{D_l}$$
  ($m_{D_l}$은 평균, $var_{D_l}$은 표준편차)

- **전략 2: $\text{SD}_{DL}$ (Minimum Distance After Length)**
  $\text{SD}_{OL}$의 결과에 최소 시맨틱 거리(MSD)를 결합하여 보정한다. MSD는 다음과 같이 정의된다.
  $$\text{MSD}(x) = \min_{z \in Z_s} \|F(x) - z\|_2^2$$
  $\text{SD}_{OL}$이 Seen으로 판별했더라도 $\text{MSD}(x) \ge R_0$이면 Unseen으로 수정하고, Unseen으로 판별했더라도 $\text{MSD}(x) < R_1$이면 Seen으로 수정한다. 여기서 $R_0$와 $R_1$은 각각 MSD의 평균에 표준편차의 2배와 1배를 더한 값이다.

- **전략 3: $\text{SD}_{WS}$ (Weighted Sum)**
  노름 차이와 MSD의 가중 합을 이용해 판별한다.
  $$\text{SD}_{WS}(x) = \begin{cases} S, & D_l(x) + \lambda \text{MSD}(x) < R_{WS} \\ U, & D_l(x) + \lambda \text{MSD}(x) \ge R_{WS} \end{cases}$$
  여기서 $\lambda=1$로 설정하며, $R_{WS}$는 가중 합의 평균과 표준편차를 이용해 계산한다.

#### 3. 분류 모듈

- **FSC**: SD가 Seen으로 판별한 인스턴스 $x$에 대해, $\text{argmax}$를 통해 가장 확률이 높은 Seen class의 시맨틱 임베딩을 예측한다.
- **ZSL**: SD가 Unseen으로 판별한 인스턴스 $x$에 대해, 기존 ZSL 방법론을 적용하여 Unseen class 중 하나로 분류한다.

## 📊 Results

### 실험 설정

- **데이터셋**: AWA (Animals with Attributes), CUB (Birds), aPY (Pascal-Yahoo), SUN (Scenes) 총 4종의 데이터셋을 사용하였다.
- **평가 지표**: Seen class의 정확도($\text{acc}_S$)와 Unseen class의 정확도($\text{acc}_U$)를 모두 고려하여, 이들의 조화 평균인 Harmonic Mean ($H_{acc}$)을 사용한다.
  $$H_{acc} = \frac{2 \times \text{acc}_S \times \text{acc}_U}{\text{acc}_S + \text{acc}_U}$$

### 결과 분석

제시된 텍스트 내에 구체적인 수치 결과 표는 포함되어 있지 않으나, 저자는 제안한 SD 방법론이 기존의 복잡한 파라미터 설정 없이도 도메인 판별 성능을 높여 GZSL의 전체적인 정확도를 향상시켰음을 주장한다. 특히 $\text{SD}_{DL}$ 전략이 가장 좋은 성능을 보이며, $\text{SD}_{OL}$은 가장 빠르고 단순한 방법임을 명시하고 있다.

## 🧠 Insights & Discussion

### 강점

- **단순성 및 효율성**: 복잡한 신경망을 추가로 학습시키거나 하이퍼파라미터를 수동으로 튜닝할 필요 없이, 데이터의 통계적 특성(평균, 표준편차)만으로 임계값을 설정하는 적응적(Adaptive) 방식을 취했다.
- **범용성**: 특정 ZSL 모델에 종속되지 않고, 어떤 기존 ZSL 모델 및 Fully Supervised 모델과도 결합하여 GZSL 시스템을 구축할 수 있는 플러그인 형태의 구조를 가진다.

### 한계 및 논의사항

- **SM 모델 의존성**: SD의 성능이 전적으로 Semantic Mapping (SM) 모듈이 얼마나 정확하게 시각적 특징을 시맨틱 공간으로 투영하느냐에 달려 있다. 만약 SM 모듈의 성능이 낮다면 노름이나 거리 기반의 판별력이 떨어질 가능성이 크다.
- **분포 가정**: 임계값을 설정할 때 Seen class의 $D_l$ 값이 정규 분포를 따른다고 가정하고 있다. 실제 데이터에서 이 가정이 항상 유효한지에 대한 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 GZSL에서 발생하는 Seen/Unseen 도메인 오분류 문제를 해결하기 위해, 시맨틱 공간에서의 **벡터 노름(Norm) 길이와 최소 거리(MSD)**라는 기하학적 특성을 이용한 **Semantic Discriminator (SD)**를 제안한다. 이 판별기를 통해 인스턴스를 먼저 도메인별로 분류한 뒤 각각 최적화된 분류기에 전달함으로써, 복잡한 설정 없이도 GZSL의 성능을 효과적으로 개선할 수 있음을 보여준다. 이는 향후 도메인 적응형 Zero-Shot Learning 연구에 있어 단순하면서도 강력한 베이스라인으로 활용될 가능성이 높다.
