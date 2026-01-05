# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

Alex Kendall, Yarin Gal

## 🧩 Problem to Solve

오늘날 딥러닝 모델은 컴퓨터 비전 분야에서 놀라운 성능을 보여주지만, 예측의 정확성에 대한 불확실성을 정량화하지 못하는 경우가 많습니다. 이러한 "맹목적인" 예측은 자율 주행 시스템의 사고나 인종 차별 논란 같은 심각한 결과를 초래할 수 있습니다. 기존 딥러닝 모델은 회귀 문제에서 불확실성을 표현하거나 분류 모델에서 모델 불확실성을 포착하는 데 어려움이 있습니다. 베이지안 딥러닝을 통해 이러한 불확실성을 모델링할 수 있지만, 데이터에 내재된 노이즈를 포착하는 **Aleatoric uncertainty**와 모델 파라미터의 불확실성을 포착하는 **Epistemic uncertainty** 중 어떤 불확실성이 컴퓨터 비전 작업에 더 중요하고 효과적인지 명확히 이해해야 합니다.

## ✨ Key Contributions

- **정확한 불확실성 포착**: 특히 분류 작업에 대한 새로운 접근 방식을 통해 Aleatoric 및 Epistemic uncertainty를 정확하게 포착하는 방법을 제시했습니다.
- **성능 향상**: Aleatoric uncertainty의 명시적 모델링을 통해 학습된 손실 감쇠(attenuation)를 유도하여 노이즈 데이터의 영향을 줄임으로써 비-베이지안 모델 대비 1-3%의 모델 성능 향상을 달성했습니다.
- **불확실성 유형 비교 분석**: Aleatoric 및 Epistemic uncertainty 모델링 간의 장단점을 분석하고, 각 불확실성의 특성 및 모델 성능과 추론 시간 간의 균형을 연구했습니다.
- **통합 프레임워크 제시**: 입력 의존적인 Aleatoric uncertainty와 Epistemic uncertainty를 결합하는 베이지안 딥러닝 프레임워크를 제안했습니다.
- **최첨단 결과 달성**: 제안된 방법론이 의미론적 분할 및 깊이 회귀 벤치마크에서 새로운 최첨단(State-of-the-Art) 결과를 달성했습니다.

## 📎 Related Works

- **기존 베이지안 딥러닝**: 기존 베이지안 딥러닝 접근 방식은 주로 Epistemic uncertainty 단독 또는 Aleatoric uncertainty 단독으로 모델링합니다.
- **Epistemic Uncertainty**: 베이지안 신경망(BNN)에서 가중치에 사전 분포를 설정하고, Dropout variational inference(Monte Carlo dropout)와 같은 근사 추론 기법을 통해 가중치 사후 분포를 추론하여 모델 파라미터의 불확실성을 포착합니다.
- **Heteroscedastic Aleatoric Uncertainty**: 비-베이지안 신경망에서 입력 $x$에 따라 달라지는 관측 노이즈 $\sigma(x)^2$를 학습하여 Aleatoric uncertainty를 모델링하는 Heteroscedastic regression 방식이 존재합니다. 이 방식은 모델 파라미터 불확실성을 포착하지 않습니다.
- **기존 컴퓨터 비전 모델**: SegNet, FCN-8, DeepLab, DenseNet 등 다양한 딥러닝 아키텍처가 컴퓨터 비전 작업의 기준선으로 참조되었습니다.

## 🛠️ Methodology

저자들은 입력 의존적인 Aleatoric uncertainty와 Epistemic uncertainty를 결합하는 통합 베이지안 딥러닝 프레임워크를 제안합니다.

1. **Epistemic Uncertainty (모델 불확실성) 모델링**:

   - Dropout variational inference를 사용하며, 테스트 시 Monte Carlo dropout을 통해 모델 가중치 $\hat{W}_t$를 샘플링합니다.
   - 회귀의 경우 예측 분산(예측 평균 $\mathbb{E}(y)$와 예측 분산 $\text{Var}(y)$)을, 분류의 경우 확률 벡터의 엔트로피 $\text{H}(p)$를 통해 불확실성을 정량화합니다.

2. **Heteroscedastic Aleatoric Uncertainty (데이터 불확실성) 모델링**:

   - **회귀**: 신경망이 예측 평균 $\hat{y}$뿐만 아니라 예측 분산의 로그 값 $s = \log \hat{\sigma}^2$도 출력하도록 합니다.
   - 손실 함수는 다음과 같습니다:
     $$ \mathcal{L}_{\text{BNN}}(\theta) = \frac{1}{D} \sum_{i} \left( \frac{1}{2} \exp(-s_i) \|y_i - \hat{y}\_i\|^2 + \frac{1}{2} s_i \right) $$
        여기서 $\exp(-s_i)$ 항은 학습된 손실 감쇠(attenuation) 역할을 하여 노이즈가 많은 데이터의 영향을 줄입니다.
   - **분류**: 로짓 공간에 가우시안 분포를 적용하여 Heteroscedastic uncertainty를 확장합니다. 신경망은 각 픽셀 $i$에 대한 로짓 $f^W_i$와 로짓 분산 $(\sigma^W_i)^2$을 예측합니다.
   - 몬테카를로 적분을 통해 손실 함수를 근사하며, 이 역시 학습된 손실 감쇠 효과를 가집니다.

3. **두 불확실성의 결합**:

   - 회귀 모델의 총 예측 불확실성은 Epistemic uncertainty와 Aleatoric uncertainty의 합으로 근사됩니다:
     $$ \text{Var}(y) \approx \frac{1}{T} \sum*{t=1}^T \hat{y}\_t^2 - \left(\frac{1}{T} \sum*{t=1}^T \hat{y}_t\right)^2 + \frac{1}{T} \sum_{t=1}^T \hat{\sigma}\_t^2 $$
     여기서 첫 번째 항은 Epistemic uncertainty, 두 번째 항은 Aleatoric uncertainty에 해당합니다.

4. **구현**: DenseNet 아키텍처를 기반으로 하며, Adam optimizer와 RMS-Prop을 사용하여 훈련했습니다.

## 📊 Results

- **의미론적 분할 (Semantic Segmentation)**:
  - CamVid 데이터셋에서 67.5%의 평균 IoU, NYUv2 데이터셋에서 70.6%의 정확도 및 37.3%의 IoU를 달성하여 새로운 SOTA 기록을 세웠습니다.
  - Aleatoric uncertainty 모델링이 Epistemic uncertainty 모델링보다 더 큰 성능 향상을 보였으며, 두 불확실성을 결합했을 때 가장 좋은 성능을 달성했습니다.
- **픽셀 단위 깊이 회귀 (Pixel-wise Depth Regression)**:
  - Make3D 및 NYUv2 Depth 데이터셋에서 SOTA 결과를 달성했습니다.
  - Aleatoric uncertainty는 깊이가 큰 영역, 반사 표면, 가려짐 경계와 같이 본질적으로 어려운 부분에서 높은 불확실성을 포착했습니다.
  - Epistemic uncertainty는 훈련 데이터셋에 희귀한 객체(예: 사람)와 같이 데이터 부족으로 인한 어려움을 포착했습니다.
- **불확실성 측정의 품질**:
  - 정밀도-재현율(Precision-Recall) 곡선은 불확실성이 증가함에 따라 정밀도가 감소하여, 불확실성 측정값이 정확도와 잘 상관관계를 가짐을 보여주었습니다.
  - 보정(Calibration) 플롯은 Aleatoric 및 Epistemic uncertainty 모델링이 보정 오류(MSE)를 줄여 불확실성 예측의 품질을 향상시켰음을 보여주었습니다.
- **훈련 데이터와의 거리**:
  - 훈련 데이터셋 크기가 증가해도 Aleatoric uncertainty는 비교적 일정하게 유지되어, 데이터 자체에 내재된 노이즈를 포착함을 확인했습니다.
  - Epistemic uncertainty는 훈련 데이터셋 크기가 커짐에 따라 감소하고(설명 가능), 훈련 분포에서 멀리 떨어진(out-of-distribution) 테스트 데이터에 대해서는 크게 증가했습니다.
- **실시간 적용**:
  - Aleatoric uncertainty 모델은 계산 비용을 거의 추가하지 않습니다.
  - Epistemic uncertainty 모델(Monte Carlo dropout)은 아키텍처 전체를 샘플링해야 하는 경우 50배의 속도 저하를 야기할 수 있어 실시간 적용에 제약이 있습니다.

## 🧠 Insights & Discussion

- **상호 보완적인 역할**: Aleatoric 및 Epistemic uncertainty는 상호 보완적이며, 함께 모델링할 때 예측 성능과 견고성을 가장 크게 향상시킵니다.
- **Aleatoric uncertainty의 중요성**:
  - 데이터 양이 많은 상황에서는 Epistemic uncertainty가 대부분 설명될 수 있으므로 Aleatoric uncertainty 모델링이 더 중요합니다.
  - 실시간 애플리케이션에서는 Monte Carlo 샘플링의 높은 비용 없이 모델링할 수 있으므로 Aleatoric uncertainty가 효율적입니다.
  - 학습된 손실 감쇠를 통해 노이즈에 대한 모델의 견고성을 높입니다.
- **Epistemic uncertainty의 중요성**:
  - 안전이 중요한 시스템에서는 모델이 이전에 보지 못한 상황(out-of-distribution 데이터)을 감지하는 데 Epistemic uncertainty가 필수적입니다.
  - 훈련 데이터가 희소한 작은 데이터셋에서는 Epistemic uncertainty가 더 중요합니다.
- **미래 연구**: 딥러닝에서 실시간으로 Epistemic uncertainty를 추론하는 방법을 찾는 것이 중요한 연구 방향임을 지적합니다. 이는 자율 주행과 같이 안전에 민감한 시스템에서 맹목적인 예측으로 인한 재앙을 피하는 데 필수적입니다.

## 📌 TL;DR

**문제**: 컴퓨터 비전용 딥러닝 모델은 예측의 불확실성을 정량화하지 못하여 안전이 중요한 애플리케이션에서 치명적인 오류를 유발할 수 있습니다. 특히 데이터 노이즈(Aleatoric)와 모델 파라미터 불확실성(Epistemic) 중 어떤 불확실성이 필요한지 불분명합니다.

**제안 방법**: 입력 의존적 Aleatoric uncertainty(데이터에 내재된 노이즈 포착 및 학습된 손실 감쇠 적용)와 Epistemic uncertainty(모델 파라미터 불확실성 포착)를 결합하는 새로운 베이지안 딥러닝 프레임워크를 제시했습니다. Aleatoric uncertainty는 신경망이 예측과 함께 분산을 출력하도록 하여 손실 함수에 학습된 감쇠 인자를 포함하고, Epistemic uncertainty는 Monte Carlo dropout을 통해 근사합니다.

**핵심 결과**: 두 불확실성을 모두 모델링할 때 의미론적 분할 및 깊이 회귀 작업에서 최첨단 성능을 달성하며 모델의 정확도와 견고성을 향상시켰습니다. Aleatoric uncertainty는 대규모 데이터 및 실시간 애플리케이션에 효율적이고 노이즈에 강하며, Epistemic uncertainty는 훈련 데이터와 다른 외부 데이터(out-of-distribution) 감지 및 안전에 중요한 애플리케이션에 필수적임을 보였습니다.
