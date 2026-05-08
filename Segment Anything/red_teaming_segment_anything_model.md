# Red-Teaming Segment Anything Model

Krzysztof Jankowski, Bartlomiej Sobieski, Mateusz Kwiatkowski, Jakub Szulc, Michał Janik, Hubert Baniecki, Przemysław Biecek (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야의 대표적인 Foundation Model인 Segment Anything Model (SAM)의 취약성을 분석하기 위한 다각적인 Red-Teaming 분석을 수행한다.

SAM은 방대한 데이터로 사전 학습되어 다양한 세그멘테이션 작업에서 뛰어난 성능을 보이지만, 실제 환경에 배포하기 전에는 모델의 한계점과 잠재적 위험 요소를 깊이 있게 이해하는 것이 필수적이다. 특히 자율 주행과 같은 안전 필수 시스템(Safety-critical systems)에 SAM을 그대로 적용할 경우, 예상치 못한 환경 변화나 악의적인 공격으로 인해 치명적인 오류가 발생할 수 있다. 따라서 본 연구의 목표는 SAM의 강건성(Robustness), 프라이버시(Privacy), 그리고 적대적 공격(Adversarial Attacks)에 대한 취약성을 정밀하게 평가하여 모델의 안전성 수준을 진단하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 SAM을 대상으로 한 세 가지 관점의 Red-Teaming 분석 체계를 제안하고 실행한 것이다.

1. **스타일 변환 기반의 강건성 검증**: 실제 도로 상황에서 발생할 수 있는 다양한 기상 조건(눈, 비, 밤 등)을 시뮬레이션하여 SAM의 세그멘테이션 마스크가 어떻게 왜곡되는지 분석하였다.
2. **프라이버시 침해 가능성 평가**: 텍스트 프롬프트를 통해 특정 유명인의 얼굴을 식별하고 세그멘테이션 할 수 있는지를 테스트하여, 모델이 의도치 않은 개인 식별 지식을 보유하고 있는지 확인하였다.
3. **적대적 공격 및 새로운 공격 기법 제안**: 기존의 White-box 및 Black-box 공격의 효용성을 평가하고, 연산 효율성과 은닉성을 동시에 잡은 새로운 하이브리드 공격 알고리즘인 Focused Iterative Gradient Attack (FIGA)를 제안하였다.

## 📎 Related Works

SAM은 Vision Transformer (ViT) 기반의 Image Encoder, Prompt Encoder, Mask Decoder로 구성되어 포인트, 박스, 텍스트 등의 프롬프트를 통해 마스크를 생성한다.

기존 연구들은 SAM의 강건성을 분석하기 위해 가우시안 블러(Gaussian blur)나 색수차(Chromatic aberration)와 같은 이미지 증강(Augmentation) 기법을 사용하였다. 하지만 저자들은 이러한 왜곡들이 실제 환경에서 발생할 가능성이 낮다는 점을 지적하며, 더 현실적인 시나리오인 스타일 변환(Style transfer) 관점의 분석이 필요함을 강조한다. 또한, SAM이 형태(Shape)보다는 질감(Texture)에 더 편향되어 있다는 선행 연구를 바탕으로, 질감 변화가 심한 기상 조건에서의 취약성을 탐구한다.

적대적 공격 측면에서는 FGSM, JSMA와 같은 고전적인 White-box 공격과 SIMBA, EBAD와 같은 Black-box 공격들이 SAM에 적용된 바 있으나, 본 논문은 이를 확장하여 더 효율적인 공격 방법론을 모색한다.

## 🛠️ Methodology

### 1. 스타일 변환 강건성 분석 (Robustness to Style Transfer)

현실적인 도로 환경을 시뮬레이션하기 위해 Multi-weather-city 데이터셋을 사용한다. 분석 파이프라인은 다음과 같다.

- **단계 1**: 원본 이미지에서 SAM의 자동 마스크 생성 모드를 사용하여 마스크를 생성하고, 그중 가장 큰 $k$개의 마스크를 선정하여 각 마스크의 중심점(Center point) 좌표를 추출한다. 이 좌표들은 이후 비교를 위한 마스크 ID 역할을 한다.
- **단계 2**: 동일한 좌표를 프롬프트로 사용하여, 스타일이 변환된(기상 조건이 적용된) 이미지에서 마스크를 생성한다.
- **평가**: 원본 마스크와 변환 후 마스크 간의 mean Intersection over Union (IoU)를 계산하여 성능 저하 정도를 측정한다.

### 2. 프라이버시 공격 분석 (Robustness to Attacks on Privacy)

SAM은 기본적으로 텍스트 프롬프트를 직접 지원하지 않으므로, GroundingDINO를 텍스트 인코더로 사용하는 LangSAM 구현체를 활용한다.

- **실험 설정**: CelebA 데이터셋에서 선정된 16명의 유명인 이미지를 사용한다. 모델이 단순히 이미지 내 유일한 사람을 찾는 '치팅'을 방지하기 위해, 9명의 서로 다른 유명인 얼굴을 $3 \times 3$ 그리드 형태로 배치한 합성 이미지를 입력으로 사용한다.
- **절차**: 특정 유명인의 이름을 텍스트 프롬프트로 입력하고, 모델이 해당 인물의 위치를 정확히 세그멘테이션 하는지 확인한다.
- **평가**: 이를 이진 분류 문제로 정의하여 Precision, Recall, F1 score를 측정한다.

### 3. 적대적 공격 분석 (Robustness to Adversarial Attacks)

적대적 공격의 일반적인 최적화 목표는 다음과 같다.
$$\min_{\delta} -\|f(x)-f(x+\delta)\| + \lambda \cdot \|\delta\|$$
여기서 $\delta$는 이미지 $x$에 추가되는 섭동(Perturbation)이며, 예측 결과의 차이는 최대화하고 섭동의 크기는 최소화하는 것이 목표이다.

#### 3.1 White-box 공격: FGSM* 및 FIGA

- **FGSM***: 기존 FGSM을 세그멘테이션 작업에 맞게 수정하였다. 손실 함수를 마스크 간의 거리로 정의하고 다음과 같이 픽셀을 업데이트한다.
$$X := X - \epsilon \cdot \text{sgn}\left(\nabla_X \|\sigma(\text{SAM}(p, X)) - Y\|_2^2\right)$$
여기서 $Y$는 타겟 마스크, $p$는 텍스트 프롬프트이다.
- **FIGA (Focused Iterative Gradient Attack)**: FGSM의 광범위한 변화와 JSMA의 느린 속도를 보완한 기법이다. Saliency map을 통해 영향력이 큰 상위 $k$개의 픽셀만 선택하여 $\epsilon$만큼 업데이트한다. $k=1$이면 JSMA와 같고, $k$가 전체 픽셀 수이면 FGSM과 동일해진다.

#### 3.2 Black-box 공격

- **SIMBA**: 랜덤 직교 벡터를 이용해 반복적으로 픽셀을 섭동시키며, 고주파 노이즈를 제거하기 위해 이산 코사인 변환(DCT)을 적용한다.
- **EBAD**: 여러 대리 모델(Surrogate models)을 사용하여 생성한 섭동을 타겟 모델로 전이(Transfer)시키는 방식을 사용한다.

## 📊 Results

### 1. 스타일 변환 결과

기상 조건이 악화될수록 mean IoU가 급격히 감소하는 경향을 보였다.

- **눈(Snow)**: $0.87$, **밤(Night)**: $0.80$, **젖은 도로(Wet)**: $0.70$
- **물방울(Drops) 추가 시**: 특히 유리창에 물방울이 맺힌 경우(Night + Drops) IoU가 $0.39$까지 급락하였다.
- **분석**: 색상 변화(밤, 눈)보다 질감 변화(물방울)가 마스크 생성에 훨씬 치명적이며, 이는 SAM이 형태보다 질감에 편향되어 있다는 사실을 뒷받침한다.

### 2. 프라이버시 결과

유명인에 따라 식별 성능의 편차가 매우 크게 나타났다.

- **높은 성능**: Prince William, Paul Burrell 등 일부 인물에 대해서는 높은 F1 score를 기록하여, 모델이 특정 인물에 대한 정보를 내재하고 있음을 보였다.
- **낮은 성능**: Keanu Reeves의 경우 랜덤 분류기보다 낮은 성능을 보였다.
- **이상 현상**: Michael Jackson과 같은 경우 모든 이미지를 타겟으로 예측하는 경향을 보였다.

### 3. 적대적 공격 결과

- **White-box**: FIGA는 FGSM*보다 더 적은 섭동(MSE 기준)으로 유사하거나 더 우수한 공격 성능을 보였으며, JSMA보다 실행 속도가 훨씬 빨랐다. 또한, FGSM 계열의 공격은 가우시안 노이즈를 추가해도 공격 효과가 유지되는 강한 강건성을 보였다.
- **Black-box**: SIMBA와 EBAD 모두 SAM의 복잡한 구조로 인해 성공적인 공격을 수행하지 못했다. 특히 EBAD는 IoU가 $96\%$ 수준으로 유지되어 거의 영향이 없었다. 이는 대리 모델과 SAM 간의 파라미터 규모 차이(ResNet101 $\approx 44.5$M vs SAM $\approx 636$M)에서 기인한 것으로 추정된다.

## 🧠 Insights & Discussion

본 연구는 SAM이 범용적인 성능에도 불구하고 실제 환경에서 매우 취약할 수 있음을 입증하였다.

- **질감 편향성**: SAM은 객체의 기하학적 형태보다는 표면 질감에 의존하여 세그멘테이션을 수행하는 경향이 있다. 이로 인해 물방울과 같은 단순한 질감 왜곡에도 마스크가 완전히 파괴되는 결과가 초래된다.
- **프라이버시 위험**: LangSAM을 통한 테스트 결과, 모델이 학습 데이터셋의 특정 유명인 정보를 암기(Memorization)하고 있음을 확인하였다. 이는 향후 개인정보 보호 관점에서 심각한 문제가 될 수 있다.
- **공격의 비대칭성**: White-box 공격에는 매우 취약하지만, Black-box 공격에는 상대적으로 강건하다. 이는 모델의 내부 파라미터에 접근할 수 있을 때 생성되는 정밀한 섭동이 매우 치명적임을 의미한다.

저자들은 이에 대한 방어 전략으로 **특수 증강 데이터를 이용한 파인튜닝**, **학습 데이터 필터링 및 Unlearning 기법 적용**, 그리고 **적대적 훈련(Adversarial Training)**을 제안한다.

## 📌 TL;DR

본 논문은 SAM의 취약성을 분석하기 위해 스타일 변환, 프라이버시 침해, 적대적 공격이라는 세 가지 관점에서 Red-Teaming을 수행하였다. 연구 결과, SAM은 유리창의 물방울과 같은 질감 변화에 취약하며, 일부 유명인의 신원을 식별할 수 있는 프라이버시 위험이 있고, White-box 적대적 공격(특히 제안된 FIGA 기법)에 매우 취약함이 드러났다. 이 연구는 SAM과 같은 거대 기초 모델을 실제 시스템에 적용하기 전, 반드시 정밀한 안전성 검증과 방어 기제 구축이 필요함을 시사한다.
