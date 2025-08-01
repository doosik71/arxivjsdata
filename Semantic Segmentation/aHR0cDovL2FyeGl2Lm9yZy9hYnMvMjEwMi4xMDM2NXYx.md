# Analyzing Overfitting under Class Imbalance in Neural Networks for Image\n  Segmentation

Zeju Li, Konstantinos Kamnitsas and Ben Glocker

## 🧩 Problem to Solve

이미지 분할 분야에서 신경망은 종종 훈련 세트에 매우 적게 표현되는(under-represented) 작은 구조의 전경(foreground) 샘플에 과적합(overfitting)되는 경향이 있습니다. 이는 제한된 데이터와 심한 클래스 불균형 상황에서 모델의 일반화(generalization) 성능을 저하시켜, 테스트 시 소수 클래스의 **과소 분할(under-segmentation)**로 이어집니다. 기존의 클래스 불균형 해결책이나 과적합 완화 기법들은 이러한 신경망의 비대칭적 로짓(logit) 이동(shift) 현상을 명시적으로 고려하지 못했습니다.

## ✨ Key Contributions

* **비대칭적 로짓 분포 이동 현상 식별**: 클래스 불균형 하의 과적합이 발생할 때, 소수 클래스의 보이지 않는(unseen) 테스트 샘플에 대한 로짓 활성화가 결정 경계($$ \text{decision boundary} $$) 쪽으로, 심지어 경계를 넘어 이동한다는 사실을 경험적으로 발견했습니다. 반면, 다수 클래스 샘플은 영향을 받지 않습니다.
* **일관된 현상 확인**: 이러한 비대칭적 로짓 이동 및 이로 인한 낮은 민감도(sensitivity)는 다양한 데이터셋(BRATS, ATLAS, KiTS, 복부 장기), 태스크(뇌종양, 뇌졸중 병변, 신장 종양, 소수 장기 분할) 및 신경망 아키텍처(DeepMedic, 3D U-Net)에서 일관되게 관찰됩니다.
* **비대칭 학습 기법 제안**: 관찰된 로짓 이동을 명시적으로 해결하기 위해, 기존의 손실 함수 및 정규화 기법(예: 대규모 마진 손실, 포컬 손실, 적대적 학습, 믹스업, 데이터 증강)의 **비대칭(asymmetric) 변형**을 제안했습니다. 이들은 소수 클래스의 로짓 활성화를 결정 경계에서 멀리 유지하도록 설계되었습니다.
* **성능 향상 입증**: 제안된 비대칭 기법들이 기존의 기준선 및 다른 접근 방식에 비해 분할 정확도(특히 민감도)를 크게 향상시킴을 광범위한 실험을 통해 입증했습니다.

## 📎 Related Works

* **클래스 불균형 처리**:
  * **재가중치(Re-weighting)**: 클래스 수준(예: 샘플 빈도 기반 [41], [46] 또는 고급 규칙 [9]) 및 샘플 수준(예: 하드 샘플 마이닝 [11], 포컬 손실 [29], 마진 기반 손실 [10], [31], [26], [17], [23], [5], [49]). 본 연구는 포컬 손실과 마진 손실을 소수 클래스에 유리하게 비대칭적으로 수정합니다.
  * **데이터 합성(Data Synthesis)**: SMOTE [6]와 같은 방법으로 소수 클래스 샘플을 생성. 본 연구는 잠재 특징 공간(latent feature space)에서 샘플을 합성(믹스업, 적대적 학습)하거나 비대칭적 데이터 증강을 제안합니다.
  * **기타 방법**: 훈련 전략 변경(예: 출력 레이어 재훈련 [15]), 테스트 시 결정 경계 조정 [21], [38], [48], 메타 학습 [42], 전이 학습 [32].
  * **분할 특화**: 경계 손실(boundary loss) [22], 다단계 접근법 [36], [39].
* **일반적인 정규화 기법**: 드롭아웃 [37], 가중치 감소(weight decay) [24], 데이터 증강 [7], [8], 데이터 믹싱(믹스업) [45], [47], 적대적 학습 [12], [44]. 대부분은 클래스 불균형을 명시적으로 다루지 않습니다.

## 🛠️ Methodology

본 연구는 신경망의 편향된(biased) 동작을 바탕으로, 소수 클래스(전경)의 로짓 활성화를 결정 경계에서 멀리 유지하기 위해 기존 손실 함수 및 훈련 전략에 다음과 같은 비대칭적 수정 사항을 적용합니다.

* **비대칭 대규모 마진 손실 (Asymmetric Large Margin Loss)**:
  * 클래스 간 로짓의 유클리드 거리($$ \text{Euclidean distance} $$)를 늘리는 대규모 마진 손실 [40]에서, 소수 클래스($$ r_j = 1 $$)에만 마진 $$ m $$을 적용하여 결정 경계를 소수 클래스에 유리하게 이동시킵니다.
  * 확률 $$ \hat{q}_{ij} $$ 계산: $$ \hat{q}_{ij} = \frac{e^{z_{ij} - y_{ij} r_j m}}{\sum_{k=1}^{c} e^{z_{ik} - y_{ik} r_k m}} $$
* **비대칭 포컬 손실 (Asymmetric Focal Loss)**:
  * 잘 분류된 샘플의 가중치를 줄이는 포컬 손실 [29]의 문제를 해결하기 위해, 전경 클래스($$ r_j = 1 $$)에 대한 손실 감쇠(attenuation)를 제거하여 로짓이 결정 경계에 너무 가까워지는 것을 방지합니다.
  * 수정된 크로스 엔트로피($$ \text{CE} $$) 포컬 손실: $$ \hat{L}_{CE}^{\text{focal}}(x_i, y_i) = \sum_{j=1}^{c} \left( -r_j y_{ij} \log(p_{ij}) - (1-r_j)(1-p_{ij})^\gamma y_{ij} \log(p_{ij}) \right) $$
  * 다이스 유사도 계수($$ \text{DSC} $$) 손실에 대한 비대칭 포컬 변형도 제안됩니다.
* **비대칭 적대적 학습 (Asymmetric Adversarial Training)**:
  * 적대적 샘플을 생성하여 모델 견고성(robustness)을 높이는 적대적 학습 [12]에서, 소수 클래스($$ y_i \cdot r > 0 $$)에 대한 적대적 섭동($$ \text{perturbation} $$)을 생성하는 데 더 집중합니다.
  * 수정된 적대적 방향 $$ \hat{d}_{\text{adv}} $$: $$ \hat{d}_{\text{adv}} = \text{argmax}_{d;\|d\|<\epsilon} L(x_i+d, y_i \odot r) \text{ s.t. } y_i \cdot r > 0 $$
* **비대칭 믹스업 (Asymmetric Mixup)**:
  * 이미지 및 레이블 쌍의 선형 조합으로 훈련 샘플을 생성하는 믹스업 [47]에서, 배경 클래스에서 특정 거리 이상 떨어진 혼합 이미지($$ \tilde{x}_i $$)를 전경 샘플로 간주하여 강성 레이블(hard labels)을 생성합니다.
* **비대칭 증강 (Asymmetric Augmentation)**:
  * 표준 데이터 증강과 달리, 배경 클래스에 대한 변환 샘플 수를 줄이고 전경 클래스($$ y_i \cdot r == 1 $$)에 대해 더 강한 변환($$ A(x_i) $$)을 적용합니다.
* **비대칭 기법의 조합**: 위에 언급된 여러 비대칭 기법들을 단일 모델에 통합하여 시너지 효과를 추구할 수 있습니다.

## 📊 Results

* **일관된 현상 확인**: BRATS (뇌종양), ATLAS (뇌졸중 병변), KiTS (신장 종양), 복부 장기 분할 등 4가지 데이터셋과 DeepMedic, 3D U-Net 등 2가지 아키텍처에서 실험한 결과, 훈련 데이터가 적을수록 테스트 데이터의 민감도가 크게 감소하며 과소 분할이 발생함이 확인되었습니다.
* **로짓 분포 변화 시각화**: 제안된 비대칭 기법들은 소수 클래스(전경)의 로짓 분포를 확장하고, 보이지 않는 전경 샘플의 로짓이 결정 경계의 올바른 쪽에 머무르도록 유도하여 로짓 이동을 효과적으로 완화합니다.
* **성능 개선**:
  * **비대칭 대규모 마진 손실** 및 **비대칭 포컬 손실**은 대칭 버전에 비해 모든 경우에서 분할 성능을 향상시켰습니다. 특히 포컬 손실은 소수 클래스의 손실 감쇠를 제거하여 민감도를 높였습니다.
  * **비대칭 적대적 학습**은 데이터 증강이 없는 환경에서 특히 효과적이며, 데이터 증강이 있는 경우에도 기존 적대적 학습보다 우수한 성능을 보였습니다.
  * **비대칭 믹스업**은 BRATS 데이터셋에서 큰 성능 향상을 보였으나, ATLAS 및 KiTS와 같이 이미지 채널이 하나이고 전경/배경 강도 분포가 겹치는 경우 효과가 제한적이었습니다.
  * **비대칭 증강**은 대부분의 경우 소수 클래스의 DSC와 민감도 측면에서 효과적이었습니다.
  * **비대칭 기법들의 조합**은 모든 경우에서 가장 우수하고 일관된 분할 결과를 도출하여, 민감도를 크게 향상시켰습니다. 반면 대칭 기법들의 조합은 과적합을 완화하지 못했습니다.
* **기존 방법의 한계**: 단순히 소수 클래스에 대한 가중치를 늘리거나 F-스코어 기반 손실 함수를 사용하는 등의 기존 방법들은 훈련 샘플이 제한적일 때 성능 향상에 거의 영향을 미치지 못했습니다.

## 🧠 Insights & Discussion

* **과적합의 본질 재조명**: 이 연구는 클래스 불균형 하의 신경망 과적합이 단순히 훈련 데이터에 대한 과도한 적합을 넘어, 보이지 않는 소수 클래스 샘플의 로짓 분포가 결정 경계를 향해 편향되게 이동하는 현상을 야기함을 밝혀냈습니다. 이는 모델이 소수 클래스의 복잡한 패턴을 개별적으로 암기하는 경향이 있지만, 이러한 암기된 필터가 새로운 데이터에 일반화되지 못하기 때문입니다.
* **해결책의 원리**: 제안된 비대칭적 접근 방식들은 이러한 편향된 로짓 이동을 직접적으로 해결하여, 소수 클래스의 로짓을 결정 경계에서 멀리 떨어뜨리고 분포 영역을 확장함으로써 일반화 성능을 향상시킵니다.
* **로짓 분포 분석의 중요성**: 로짓 분포를 플로팅하여 시각적으로 분석하는 것은 신경망의 동작을 이해하고 특정 훈련 시나리오(예: 도메인 이동, 자기 지도 학습)에서 발생하는 문제를 진단하는 데 유용한 도구가 될 수 있음을 시사합니다.
* **한계 및 향후 연구**: 비대칭 믹스업의 경우, 다채널 이미지 데이터에서 더 효과적이었으며, 단일 채널 및 강도 분포가 겹치는 데이터에서는 제한적인 효과를 보였습니다. 이는 혼합 샘플의 유용성이 데이터 특성에 따라 달라질 수 있음을 보여줍니다. 또한, 일부 작은 클래스의 경우 후처리(post-processing)에 의해 잘못 제거될 수 있는 문제가 관찰되었으며, 이는 향후 고급 후처리 방법의 필요성을 시사합니다.

## 📌 TL;DR

클래스 불균형이 심한 이미지 분할에서 신경망은 소수 클래스에 과적합되어 테스트 시 해당 클래스의 로짓 활성화가 결정 경계 쪽으로 편향되게 이동하며 민감도가 크게 저하됩니다. 이 연구는 이 현상을 경험적으로 규명하고, 이를 해결하기 위해 기존 손실 함수(대규모 마진, 포컬) 및 정규화 기법(적대적 학습, 믹스업, 데이터 증강)의 **비대칭 변형**들을 제안합니다. 이 비대칭 기법들은 소수 클래스의 로짓을 결정 경계에서 밀어내어 일반화 성능과 분할 민감도를 획기적으로 향상시킴을 여러 데이터셋과 아키텍처에서 입증했습니다.
