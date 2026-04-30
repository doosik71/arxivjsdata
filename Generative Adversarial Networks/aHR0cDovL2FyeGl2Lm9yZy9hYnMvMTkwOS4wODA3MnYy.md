# Adversarial Attacks and Defenses in Images, Graphs and Text: A Review

Han Xu, Yao Ma, Haochen Liu, Debayan Deb, Hui Liu, Jiliang Tang, Anil K. Jain (2019)

## 🧩 Problem to Solve

본 논문은 딥 뉴럴 네트워크(Deep Neural Networks, DNN)가 다양한 도메인에서 뛰어난 성능을 보이고 있음에도 불구하고, **Adversarial Examples(적대적 예제)**라는 취약점에 노출되어 있다는 점을 지적한다. Adversarial Examples는 공격자가 모델이 잘못된 예측을 하도록 의도적으로 설계한 입력값으로, 인간이 보기에는 원본과 거의 차이가 없으나 모델에게는 치명적인 오작동을 유발한다.

특히 자율 주행 자동차의 도로 표지판 인식이나 금융 사기 탐지 시스템과 같은 **안전 필수 애플리케이션(Safety-critical applications)**에 딥러닝을 적용할 때, 이러한 취약점은 심각한 보안 사고나 물리적 위험을 초래할 수 있다. 따라서 본 논문의 목표는 이미지, 그래프, 텍스트라는 세 가지 주요 데이터 타입에 대해 최신 적대적 공격(Attack) 알고리즘과 이에 대응하는 방어(Defense) 메커니즘을 체계적이고 종합적으로 리뷰하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥러닝 모델의 보안성을 위협하는 공격 방식과 이를 막기 위한 방어 전략을 데이터 도메인별로 분류하고 분석한 종합적인 가이드라인을 제공하는 것이다.

1.  **공격 및 방어의 체계적 분류(Taxonomy):** 공격자의 목표, 지식 수준, 피해 모델의 특성에 따라 공격 유형을 정의하고, 방어 전략을 Gradient Masking, Robust Optimization, Adversarial Detection의 세 가지 범주로 구조화하였다.
2.  **멀티 도메인 분석:** 이미지뿐만 아니라 그래프(Graph)와 텍스트(Text) 데이터의 특수성을 고려한 적대적 공격 및 방어 기법을 상세히 다루었다.
3.  **이론적 배경 및 지표 제공:** 모델의 강건성(Robustness)과 적대적 리스크(Adversarial Risk)를 정량적으로 평가하기 위한 수학적 정의를 제시하여 분석의 학술적 기초를 마련하였다.

## 📎 Related Works

논문은 적대적 예제의 존재를 처음 알린 Szegedy et al. (2013)부터 FGSM(Goodfellow et al., 2014b) 등 초기 연구들을 언급하며, DNN이 입력의 미세한 섭동(Perturbation)에 얼마나 취약한지를 설명한다.

기존 연구들은 주로 이미지 도메인에 집중되어 있었으나, 본 논문은 이를 확장하여 그래프 구조 데이터나 이산적인(Discrete) 특성을 가진 텍스트 데이터로 논의를 넓혔다. 특히 기존의 방어 기법들이 새로운 공격 기법에 의해 무력화되는 '창과 방패'의 반복적인 대립 구조(Arms race)를 분석하며, 단순한 기법 도입보다는 근본적인 강건성 확보가 필요함을 강조한다.

## 🛠️ Methodology

본 논문은 리뷰 논문으로서 특정 알고리즘을 제안하기보다, 기존의 방법론들을 체계적으로 분석한다. 주요 내용은 다음과 같다.

### 1. 보안 평가 지표 및 정의
모델의 강건성을 측정하기 위해 다음과 같은 정의를 사용한다.

*   **최소 섭동(Minimal Perturbation):** 모델 $F$가 데이터 $(x, y)$에 대해 잘못된 예측을 하게 만드는 가장 작은 크기의 섭동 $\delta_{min}$을 의미한다.
    $$\delta_{min} = \arg \min_{\delta} ||\delta|| \quad \text{subject to} \quad F(x+\delta) \neq y$$
*   **강건성(Robustness):** 위에서 정의한 최소 섭동의 노름(Norm) 값이다.
    $$r(x, F) = ||\delta_{min}||$$
*   **적대적 리스크(Adversarial Risk):** 입력 $x$의 $\epsilon$-이웃 구역 내에서 손실 함수 $L$을 최대화하는 가장 공격적인 예제 $x_{adv}$에 대한 기대 손실이다.
    $$R^{adv}(F) = \mathbb{E}_{x \sim D} \left[ \max_{||x' - x|| < \epsilon} L(\theta, x', y) \right]$$

### 2. 적대적 공격 방법론 (Images 기준)
공격은 지식 수준에 따라 White-box, Black-box, Gray-box로 나뉜다.

*   **FGSM (Fast Gradient Sign Method):** 손실 함수의 기울기(Gradient) 방향으로 한 번의 스텝만 이동하여 빠르게 적대적 예제를 생성한다.
    $$x' = x + \epsilon \operatorname{sign}(\nabla_x L(\theta, x, y))$$
*   **PGD (Projected Gradient Descent):** FGSM을 여러 번 반복 수행하며, 매 단계마다 $\epsilon$-ball 내부로 투영(Project)하여 최적의 적대적 예제를 찾는다.
*   **C&W Attack:** Cross-entropy 손실 대신 Margin loss를 사용하여, 시각적으로는 거의 변화가 없으면서도 모델을 확실히 속이는 최소 섭동을 찾는다.

### 3. 방어 방법론
*   **Gradient Masking/Obfuscation:** 기울기 정보를 숨기거나 왜곡하여 공격자가 $\nabla_x L$을 계산하지 못하게 하는 방법이다. (예: Defensive Distillation, Thermometer Encoding)
*   **Robust Optimization:** 학습 과정에서 적대적 예제를 함께 학습시키는 방법이다.
    *   **Adversarial Training:** 학습 데이터셋에 PGD 등으로 생성한 적대적 예제를 포함시켜 모델이 이를 올바르게 분류하도록 훈련한다.
    *   **Certified Defenses:** 수학적으로 특정 범위 내의 모든 섭동에 대해 안전함을 보장하는 '인증서(Certificate)'를 생성한다.
*   **Adversarial Detection:** 입력값이 들어왔을 때 이것이 정상 데이터인지 적대적 예제인지 먼저 판별하는 보조 모델을 사용한다.

### 4. 기타 도메인 확장
*   **Graphs:** 노드 특징(Feature)이나 에지(Edge) 구조를 변경한다. 특히 GCN의 경우 이웃 노드의 정보를 집계하므로, 에지 하나를 추가/삭제하는 것만으로도 주변 노드들의 예측값에 영향을 줄 수 있다.
*   **Text:** 텍스트는 이산적(Discrete) 데이터이므로 기울기를 직접 사용할 수 없다. 대신 단어 교체(Synonym replacement), 오타 삽입, 문장 구조 변경(Paraphrasing) 등의 방식을 사용한다.

## 📊 Results

본 논문은 직접적인 실험을 수행하는 대신, 기존 문헌들의 결과를 종합하여 분석한다.

1.  **공격의 효율성:** PGD와 C&W 공격은 대부분의 단순 방어 기법(Gradient Masking 등)을 무력화할 수 있을 만큼 강력하다. 특히 C&W 공격은 Defensive Distillation의 취약점을 증명하였다.
2.  **방어의 한계:** Adversarial Training은 $\ell_p$ 노름 기반 공격에 대해서는 효과적이지만, 훈련 시간이 매우 오래 걸리며(PGD 기준), 학습하지 않은 형태의 새로운 공격(예: Spatially Transformed Attack)에는 여전히 취약할 수 있다.
3.  **도메인별 특성:** 텍스트 도메인에서는 단 한 글자의 변경(HotFlip)만으로도 분류 결과가 완전히 바뀌는 현상이 관찰되었으며, 이는 딥러닝 모델이 텍스트의 의미론적 구조보다는 특정 토큰의 통계적 특성에 의존함을 시사한다.
4.  **물리적 세계의 위협:** 실제 도로 표지판에 특정 스티커를 붙이는 것만으로 자율주행 차량의 인식 모델을 속일 수 있음이 확인되어, 디지털 환경뿐만 아니라 물리적 환경에서의 보안이 시급함을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 보고서는 단순히 알고리즘을 나열하는 것에 그치지 않고, **"왜 적대적 예제가 존재하는가"**에 대한 이론적 고찰을 포함하고 있다. 모델의 결정 경계(Decision Boundary)가 너무 평평하거나(Flat), 너무 굴곡져 있거나(Curved), 혹은 유연하지 못해 발생하는 문제라는 점을 분석하였다. 또한, 한 모델에서 생성된 적대적 예제가 다른 모델에서도 작동하는 **전이성(Transferability)** 특성을 분석하여 Black-box 공격의 가능성을 논리적으로 설명하였다.

### 한계 및 비판적 해석
1.  **방어의 일시성:** 논문에서 언급하듯, 새로운 방어 기법이 나오면 이를 깨는 더 강력한 공격 기법이 즉각적으로 등장한다. 이는 현재의 방어 방식들이 모델의 근본적인 구조를 개선하기보다 특정 공격 패턴을 막는 '사후 처방'에 가깝다는 것을 의미한다.
2.  **정확도와 강건성의 트레이드-오프:** 강건성을 높이기 위한 Robust Optimization 과정에서 모델의 일반적인 정확도(Clean Accuracy)가 하락하는 경향이 있다. 이는 보안성과 성능 사이의 최적 균형점을 찾는 것이 매우 어려운 문제임을 시사한다.
3.  **데이터셋의 한계:** 많은 연구가 MNIST나 CIFAR-10 같은 소규모 데이터셋에서 이루어지고 있어, 실제 복잡한 환경에서의 일반화 가능성에 대해서는 명확한 결론을 내리기 어렵다.

## 📌 TL;DR

본 논문은 이미지, 그래프, 텍스트 도메인 전반에 걸친 **적대적 공격과 방어 기법을 총망라한 리뷰 논문**이다. 공격자는 모델의 기울기를 이용해 최소한의 변화로 오작동을 유도하며, 방어자는 학습 데이터 증강(Adversarial Training)이나 기울기 은폐(Gradient Masking) 등을 통해 이에 대응한다. 하지만 여전히 완벽한 방어책은 없으며, 보안성과 정확도 사이의 트레이드-오프가 존재한다. 이 연구는 향후 **안전 필수 시스템(Safety-critical systems)**에 딥러닝을 도입하기 위해 반드시 해결해야 할 보안 과제들을 정의했다는 점에서 매우 중요하다.