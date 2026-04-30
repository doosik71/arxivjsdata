# Visual Interpretable and Explainable Deep Learning Models for Brain Tumor MRI and COVID-19 Chest X-ray Images

Yusuf Brima and Marcellin Atemkeng (2023)

## 🧩 Problem to Solve

딥러닝은 의료 영상 분석 분야에서 뛰어난 성능을 보이고 있으나, 모델의 의사결정 과정이 불투명한 '블랙박스' 특성을 가지고 있다. 이러한 해석 가능성(Interpretability)의 결여는 안전성이 최우선인 의료 현장에서 임상 전문가들이 딥러닝 모델을 신뢰하고 채택하는 데 큰 걸림돌이 된다.

본 논문의 목표는 다양한 딥러닝 모델이 의료 영상을 어떻게 분석하는지를 시각적으로 설명하는 Attribution 기법들을 평가하는 것이다. 구체적으로 뇌종양 MRI와 COVID-19 흉부 X-ray 데이터셋을 활용하여, 모델의 예측 근거를 시각화하는 방법론이 실제 의료 도메인 전문가에게 유용한 통찰을 제공하고 신뢰도를 높일 수 있는지 분석하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 다양한 CNN 아키텍처와 의료 영상 데이터셋에 대해 적응형 경로 기반 그래디언트 통합(Adaptive path-based gradient integration) 기법을 적용하여 포괄적인 평가를 수행했다는 점이다. 

단순히 모델의 정확도를 높이는 것에 그치지 않고, 어떤 Attribution 방법이 의료 영상의 생체 지표(Biomarker)를 가장 잘 포착하는지, 그리고 모델 아키텍처의 구조가 시각적 설명 가능성에 어떤 영향을 미치는지를 분석하여 제시하였다. 특히 XRAI와 같은 최신 기법이 기존의 단순 그래디언트 기반 방법보다 의료 영상 분석에서 더 높은 해석력을 제공함을 입증하였다.

## 📎 Related Works

논문에서는 의료 영상 해석을 위한 기존의 접근 방식들을 다음과 같이 분류하고 그 한계를 지적한다.

- **Concept Learning**: 고수준의 임상 개념을 조작하여 학습시키지만, 어노테이션 비용이 매우 높고 개념과 작업 간의 불일치로 인한 정보 누출(Information leakage) 문제가 발생한다.
- **Case-Based Models (CBMs)**: 입력 이미지와 기본 템플릿 간의 유사도를 측정하여 분류하지만, 노이즈나 압축 아티팩트에 취약하며 학습이 어렵다.
- **Counter Factual Explanation**: 입력 영상을 가상으로 변형하여 반대 예측을 유도하지만, 생성된 변형 영상이 실제 의료 영상과 비교해 비현실적인 경우가 많다.
- **Internal Representation Visualization**: CNN 커널의 특징 맵을 시각화하는 방식이나, 의료 영상 설정에서 이를 직접적으로 해석하는 데 한계가 있다.

본 논문은 이러한 기존 방식들의 한계, 특히 saliency mask의 노이즈 문제를 해결하기 위해 적응형 경로 기반의 통합 그래디언트(Integrated Gradients) 기법을 사용함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 연구는 의료 영상 입력 $\rightarrow$ CNN 모델을 통한 분류 $\rightarrow$ Attribution 연산자를 통한 Saliency Map 생성의 파이프라인을 가진다. 전역적 해석을 위해 t-SNE를 이용한 잠재 공간(Latent space) 시각화를 수행하며, 국소적 해석을 위해 그래디언트 기반의 Attribution 기법을 적용한다.

### 사용된 모델 및 학습 절차
총 9가지의 표준 CNN 아키텍처(VGG16, VGG19, ResNet50, ResNet50V2, DenseNet121, Xception, InceptionV3, EfficientNetB0, InceptionResNetV2)를 사용하였다.

모델 학습은 경험적 위험 최소화(Empirical Risk Minimization)를 목표로 하며, 다음과 같은 손실 함수와 $L_2$ 정규화 항을 포함한 목적 함수를 최소화하는 방향으로 최적화된다.
$$\hat{\theta}= \arg \min_{\theta} \frac{1}{N} \sum_{m=1}^{N} L(y^m, f(x^m; \theta)) + \alpha ||\theta||_2^2$$
여기서 $\theta$는 모델 파라미터, $L$은 손실 함수, $\alpha$는 정규화 하이퍼파라미터이다. 최적화 알고리즘으로는 Momentum이 적용된 확률적 경사 하강법(SDGM)을 사용하였다.

### Attribution 방법론
본 연구의 핵심인 Integrated Gradients (IG)는 비선형 미분 가능 함수 $h$에 대해, 베이스라인 입력 $x'$ 대비 입력 $x$의 예측값에 대한 기여도를 다음과 같이 계산한다.
$$IG_i(x) = (x_i - x'_i) \int_{\alpha=0}^{\alpha=1} \frac{\partial}{\partial x_i} h(x' + \alpha(x - x')) d\alpha$$

구체적인 계산 절차는 다음과 같다:
1. 모든 픽셀이 0인 예측 중립적(Prediction-neutral) 베이스라인을 설정한다.
2. 베이스라인 $x'$와 실제 입력 $x$ 사이를 선형 보간하여 여러 단계의 $\alpha$ 지점을 생성한다.
3. 각 지점에서 모델 예측값에 대한 입력 픽셀의 그래디언트를 계산하여 특징의 중요도를 측정한다.
4. 계산된 그래디언트들을 합산(Aggregation)하여 통합된 Saliency Map을 생성한다.
5. 최종 맵을 입력 이미지 스케일에 맞춰 조정하여 시각화한다.

본 논문에서는 **Vanilla Gradient**, **Guided Integrated Gradient (GIG)**, 그리고 영역 기반의 **XRAI (eXplanation with Ranked Area Integrals)** 세 가지 기법을 비교 분석하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - 뇌종양 MRI: T1-weighted CE-MRI 3,064장 (Meningioma, Glioma, Pituitary 3종 분류).
    - COVID-19 X-ray: 19,820장 (Normal, Lung Opacity, COVID-19 3종 분류).
- **지표**: Top-1 Accuracy, F1 Score, 시각적 Saliency Map의 일치도.

### 주요 결과
1. **모델 성능**: 
    - 뇌종양 MRI 데이터셋에서는 **DenseNet121**이 98.10%의 정확도로 가장 우수한 성능을 보였다.
    - COVID-19 X-ray 데이터셋에서는 **InceptionResNetV2**가 89.0%의 정확도로 가장 높았다.
2. **Attribution 기법 비교**:
    - **XRAI**가 가장 뛰어난 설명력을 보였으며, 특히 DenseNet121 모델과 결합했을 때 전문가가 세그멘테이션한 영역과 가장 유사한 특징을 포착했다.
    - **Vanilla Gradient**와 **GIG**는 Saliency Map에 노이즈가 많아 의료적 의미를 도출하기에 부적합한 것으로 나타났다.
    - **Xception** 모델은 다른 모델들에 비해 Saliency Map의 안정성이 낮고 노이즈 수준이 높게 나타나 시각적 설명력이 가장 떨어졌다.

## 🧠 Insights & Discussion

본 연구는 모델의 높은 예측 정확도가 반드시 높은 해석 가능성으로 이어지지는 않는다는 점을 시사한다. 

**강점 및 통찰**:
- XRAI와 같은 영역 기반 기법이 픽셀 단위의 그래디언트 기법보다 의료 영상의 ROI(Region of Interest)를 훨씬 더 정확하게 짚어낼 수 있음을 보였다.
- t-SNE 시각화를 통해, 학습된 모델이 잠재 공간에서 서로 다른 클래스의 매니폴드(Manifold)를 효과적으로 분리하고 있음을 확인하였다.

**한계 및 비판적 해석**:
- CNN이 학습하는 통계적 상관관계는 인간 전문가가 시각적 자극을 처리하는 생물학적 방식과 근본적으로 다르다. CNN은 텍스처나 형태에 편향(Bias)될 가능성이 높으며, 이는 잘못된 Attribution으로 이어질 위험이 있다.
- COVID-19 X-ray 데이터셋의 경우 전문가의 세그멘테이션 마스크(Ground Truth)가 제공되지 않아, MRI 데이터셋만큼 정량적인 일치도 분석을 수행하는 데 한계가 있었다.
- 결과적으로 AI 모델의 단독 판단보다는 전문가가 개입하는 'Human-in-the-loop' 시스템이 의료 현장의 위험을 줄이는 필수적인 경로임을 시사한다.

## 📌 TL;DR

본 논문은 뇌종양 MRI와 COVID-19 X-ray 영상을 대상으로 다양한 CNN 아키텍처와 Attribution 기법(Vanilla Gradient, GIG, XRAI)의 성능 및 해석력을 분석하였다. 실험 결과, **DenseNet121**(MRI)과 **InceptionResNetV2**(X-ray)가 높은 성능을 보였으며, 시각적 설명 측면에서는 **XRAI**가 가장 신뢰할 수 있는 Saliency Map을 생성함을 확인하였다. 이 연구는 의료 AI의 블랙박스 문제를 완화하여 임상 전문가의 신뢰를 얻기 위한 기술적 근거를 제공하며, 향후 Human-in-the-loop 진단 시스템 구축에 중요한 역할을 할 것으로 기대된다.