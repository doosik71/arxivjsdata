# PASS: Test-Time Prompting to Adapt Styles and Semantic Shapes in Medical Image Segmentation

Chuyan Zhang, Hao Zheng, Xin You, Yefeng Zheng, and Yun Gu (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 서로 다른 기관이나 장비에서 수집된 데이터 간의 도메인 시프트(Domain Shift) 문제를 해결하고자 한다. 의료 영상의 경우 데이터 프라이버시 문제로 인해 소스 데이터에 접근하는 것이 어렵기 때문에, 추가적인 학습 데이터 없이 테스트 단계에서만 모델을 적응시키는 Test-Time Adaptation (TTA) 방식이 매우 중요하다.

기존의 TTA 솔루션들은 다음과 같은 두 가지 주요 한계를 가진다. 첫째, 소스 학습 단계의 수정이 필요하거나 소스 데이터의 사전 정보(Source Priors)에 의존하는 경향이 있다. 둘째, 분할 작업의 핵심인 형태 관련 시맨틱 지식(Shape-related semantic knowledge)을 충분히 활용하지 못한다. 특히, 의료 영상에서는 단순한 텍스처 스타일의 변화뿐만 아니라 해부학적 구조의 형태 변이(Shape variability)가 성능 저하의 핵심 요인으로 작용하지만, 기존 방식들은 주로 저수준(Low-level)의 스타일 변화나 출력 공간의 확률 분포 최적화에만 집중하고 있다.

따라서 본 논문의 목표는 소스 데이터에 대한 접근 없이, 테스트 시점에 입력 이미지의 스타일과 고수준의 시맨틱 형태를 동시에 적응시킬 수 있는 TTA 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 입력 공간(Input-space)과 잠재 공간(Latent-space) 모두에 적응형 프롬프트를 도입하여 스타일과 형태의 불일치를 동시에 해결하는 것이다.

1. **이중 프롬프트 전략**: 입력 영상의 스타일을 조정하는 Input Decorator와 잠재 특징의 형태 불일치를 해결하는 Cross-Attention Prompt Modulator (CAPM)를 제안하여, 저수준 스타일과 고수준 시맨틱 정보를 모두 포착한다.
2. **동적 프롬프트 생성**: 모든 샘플에 동일한 프롬프트를 적용하는 대신, 입력 데이터에 조건화된 자가 조절(Self-regulating) 비주얼 프롬프트를 생성하여 도메인 내의 변동성을 수용한다.
3. **안정적인 업데이트 전략**: 온라인 TTA 환경에서 발생할 수 있는 오차 누적과 모델 붕괴를 방지하기 위해, 교사-학생(Teacher-Student) 구조 기반의 Alternating Momentum Updating (AMU) 전략을 도입한다.

## 📎 Related Works

기존의 TTA 연구들은 크게 세 가지 방향으로 진행되어 왔다.

- **자가 지도 보조 작업**: 오토인코더(Auto-encoder) 등을 사용하여 적응을 유도하지만, 이는 사전 학습 단계의 수정이 필요하다는 단점이 있다.
- **모델 기반 TTA**: 엔트로피 최소화(Entropy Minimization)나 배치 정규화(Batch Normalization) 층의 통계값 정렬을 사용하지만, 분할 작업의 고유한 특성인 구조적 형태를 유지하는 데 한계가 있다.
- **소스 기반 TTA**: 클래스 비율이나 형태 사전 정보(Shape prior)를 사용해 정규화를 수행하지만, 소스 데이터에 대한 접근이 필요하며 분포 외(Out-of-distribution) 형태가 나타날 경우 오차가 누적될 위험이 있다.

최근의 Visual Prompt Learning은 입력 이미지에 학습 가능한 프롬프트를 추가하여 효율적인 적응을 꾀하지만, 대개 고정된 프롬프트를 사용하므로 의료 영상과 같이 데이터 다양성이 큰 도메인에서는 한계가 있다. PASS는 이러한 한계를 극복하기 위해 샘플별 맞춤형 프롬프트를 생성하는 방식을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조

PASS 프레임워크는 크게 세 가지 구성 요소로 이루어져 있다: **Input Decorator (ID)**, **Cross-Attention Prompt Modulator (CAPM)**, 그리고 **Alternating Momentum Updating (AMU)** 전략이다.

### 1. Input Decorator (ID)

ID는 입력 이미지의 스타일을 소스 도메인의 분포에 가깝게 변환하여 스타일 시프트를 줄이는 역할을 한다. 각 테스트 샘플 $x_{t_i}$에 대해 다음과 같이 변환된 이미지 $\tilde{x}_{t_i}$를 생성한다.

$$\tilde{x}_{t_i} = x_{t_i} + ID(x_{t_i}; \phi_D)$$

여기서 $ID$는 스타일 전이에 효과적인 Instance Normalization (IN)이 포함된 얕은 컨볼루션 층으로 구성되어, 각 샘플에 특화된 동적 비주얼 프롬프트를 생성한다.

### 2. Cross-Attention Prompt Modulator (CAPM)

CAPM은 잠재 공간에서 시맨틱 형태의 불일치를 해결한다.

- **Shape Prompt Bank ($SP$)**: $L$개의 학습 가능한 형태 프롬프트 템플릿을 저장하는 뱅크를 유지한다. 이는 타겟 도메인 내에서 공유된다.
- **Cross-Attention 메커니즘**: 인코더에서 추출된 타겟 특징 $z_{t_i}$를 쿼리(Query)로, $SP$를 키(Key)와 값(Value)으로 사용하여 상호작용을 수행한다.

$$Q_i = \phi_q(z_{t_i}), \quad K_i = \phi_k(SP), \quad V_i = \phi_v(SP)$$
$$CAPM(z_{t_i}, SP) = \text{Softmax}((Q_i \times K_i^T) / \sqrt{L}) V_i$$

이때 모든 템플릿을 사용하는 대신, 상위 $k$개의 어텐션 점수만을 남기는 희소 어텐션 맵 $A^*$를 사용하여 가장 관련성이 높은 형태 프롬프트만을 선택적으로 결합한다. 최종 특징 $\tilde{z}_{t_i}$는 다음과 같이 계산된다.

$$\tilde{z}_{t_i} = z_{t_i} + CAPM(V_i, A^*_i; \phi_M)$$

### 3. Alternating Momentum Updating (AMU)

온라인 TTA에서는 샘플이 순차적으로 들어오므로, 특정 샘플에 과적합되어 이전 지식을 잊어버리는 치명적 망각(Catastrophic Forgetting) 문제가 발생할 수 있다. 이를 해결하기 위해 Teacher-Student 구조를 도입한다.

- **Student 모델**: 현재 들어온 샘플 $x_{t_i}$에 대해 직접 최적화되어 빠르게 적응한다.
- **Teacher 모델**: Student 모델의 가중치를 지수 이동 평균(EMA) 방식으로 업데이트하여 역사적 지식을 저장하는 버퍼 역할을 한다.

$$\theta'_i = (1-m)\theta'_{i-1} + m\theta_i$$

특히, 본 논문에서는 고정된 모멘텀 $m$ 대신 시간이 지남에 따라 감소하는 적응형 모멘텀을 제안하여 후속 샘플이 버퍼에 주는 영향을 줄이고 학습 안정성을 높인다.

$$m_{i+1} = c + m_i \omega$$

여기서 $c$는 하한선 상수이며, $\omega$는 감쇠 계수이다. 매 새로운 샘플이 들어올 때마다 Student 모델은 Teacher 모델의 가중치로 리셋되어 기본 지식을 회복한 후 다시 적응 과정을 거친다.

## 📊 Results

### 실험 설정

- **데이터셋**: 안저 이미지의 시신경 유두/컵(Optic Disc/Cup) 분할 및 MRI 전립선(Prostate) 분할 작업.
- **평가 지표**: Dice Similarity Coefficient (DSC), 95th percentile Hausdorff Distance (HD95).
- **비교 대상**: TENT, AdaMI, ProSFDA, VPTTA 등 13가지 TTA 방법론.

### 주요 결과

1. **정량적 성능**: PASS는 오프라인 및 온라인 TTA 설정 모두에서 기존 SOTA 방법들을 상회하는 성능을 보였다. 특히 전립선 분할 작업과 같이 형태 변이가 심한 데이터셋에서 타 방법론 대비 월등한 DSC 향상을 기록하였다.
2. **온라인 TTA 안정성**: Figure 3 및 Table II/IV를 통해, PASS가 적은 파라미터 수 증가만으로도 매우 효율적인 적응 성능을 보임을 입증하였다.
3. **절제 실험(Ablation Study)**:
    - ID를 제거할 경우 전립선 데이터셋에서 성능이 크게 하락하여, 스타일 제거의 중요성이 확인되었다.
    - CAPM을 제거할 경우 시맨틱 형태 복원 능력이 떨어져 성능이 감소하였다.
    - AMU 전략이 없을 경우(Independent 또는 Continual 방식), 온라인 설정에서 오차 누적으로 인해 성능 변동성이 심해지거나 모델이 붕괴되는 현상이 관찰되었다.

## 🧠 Insights & Discussion

### 강점 및 해석

PASS는 의료 영상 분할에서 단순한 픽셀 값의 변화(Style)뿐만 아니라 구조적 형태(Shape)의 변화가 성능에 결정적인 영향을 미친다는 점을 정확히 짚어냈다. 특히 $\text{Input} \to \text{Latent}$ 공간으로 이어지는 이중 프롬프트 구조는 모델이 기존의 사전 학습된 지식을 유지하면서도 타겟 도메인의 특성에 맞게 지식을 '인출(Retrieve)'할 수 있게 한다.

### 한계 및 비판적 해석

논문에서는 Polyp segmentation 작업으로 확장 실험을 진행하였으며, 여기서 CAPM의 효과가 일부 도메인에서는 제한적임을 밝혔다. t-SNE 시각화 결과, 형태 분포가 매우 분산된(Scattered) 데이터셋의 경우, 현재의 단일 상호작용 패턴을 가진 프롬프트 뱅크 방식으로는 한계가 있음이 드러났다. 이는 모든 의료 영상에 범용적으로 적용하기 위해서는 더 유연한 동적 형태 프롬프트 생성 기법이 필요함을 시사한다.

또한, 모멘텀 감쇠 계수 $\omega$나 뱅크 크기 $L$과 같은 하이퍼파라미터에 따른 성능 변화가 존재하므로, 실제 적용 시 도메인별 최적값 탐색 과정이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 의료 영상 분할의 도메인 시프트를 해결하기 위해 스타일과 시맨틱 형태를 동시에 적응시키는 **PASS** 프레임워크를 제안한다. **Input Decorator**를 통해 저수준 스타일을 맞추고, **CAPM**과 **Shape Prompt Bank**를 통해 고수준 형태 불일치를 해결하며, **AMU** 전략으로 온라인 학습의 안정성을 확보하였다. 이 연구는 소스 데이터에 접근할 수 없는 실제 임상 환경에서 사전 학습된 모델을 효율적으로 배포하고 최적화하는 데 중요한 기여를 할 것으로 기대된다.
