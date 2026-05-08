# Shortcut Learning in Medical Image Segmentation

Manxi Lin, Nina Weng, Kamil Mikolaj, Zahra Bashir, Morten B. S. Svendsen, Martin G. Tolsgaard, Anders N. Christensen, Aasa Feragen (2024)

## 🧩 Problem to Solve

본 논문은 기계 학습 모델이 훈련 데이터셋에서 정답과 상관관계가 있지만 실제 핵심 특징은 아닌, 단순하고 오해의 소지가 있는 단서(spurious cues)를 학습하여 일반화 성능이 떨어지는 현상인 Shortcut Learning에 주목한다. 기존의 Shortcut Learning 연구는 주로 이미지 분류(Image Classification) 작업에 집중되어 있었으며, 이미지 분할(Image Segmentation) 작업에서의 위험성은 상대적으로 덜 논의되었다.

일반적으로 분할 작업은 객체의 경계를 정밀하게 묘사해야 하므로, 단순한 마커에 의존하는 분류 작업보다 Shortcut Learning에 더 강건할 것이라고 가정되어 왔다. 그러나 본 연구는 이러한 가정에 의문을 제기하며, 의료 이미지 분할에서도 Shortcut Learning이 발생할 수 있으며, 이는 특히 테스트 단계에서 객체를 인식하지 못하는 False Negative를 유발하여 임상적 정확도를 심각하게 저해할 수 있음을 입증하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 이미지 분할에서 발생하는 두 가지 서로 다른 메커니즘의 Shortcut Learning 사례를 발굴하고 이를 증명한 것이다.

1. **외부적 단서에 의한 Shortcut**: 태아 초음파 영상에서 임상적으로 추가된 텍스트나 측정용 캘리퍼(calipers)가 특정 해부학적 구조와 강하게 결합되어, 모델이 실제 해부학적 특징이 아닌 이러한 표식을 통해 분할을 수행하는 현상을 밝혔다.
2. **구조적/데이터적 결합에 의한 Shortcut**: CNN의 Zero-padding 기법과 데이터셋 구축 시의 Center-cropping 경향이 결합되어, 모델이 "이미지 경계 근처의 픽셀은 배경(background)이다"라는 잘못된 규칙을 학습하는 현상을 분석하였다.

## 📎 Related Works

기존 연구들에서 의료 이미지 내의 Shortcut으로 작용하는 변수들은 다음과 같이 보고되었다.

- **치료 및 진단 흔적**: 흉부 X-ray의 흉관(chest tube), 피부 병변의 잉크 표시나 자(rulers), 초음파의 텍스트 및 캘리퍼 등이 이에 해당한다.
- **수집 과정의 전역 정보**: 병원별 스캐너의 특성(style), 특정 병원의 질환 유병률 차이, 환자의 인구통계학적 특징(인종, 성별) 등이 Shortcut으로 작용하여 모델의 공정성과 신뢰성을 해칠 수 있다.

대부분의 선행 연구는 분류 작업에 치중되어 있으며, 본 논문은 이러한 논의를 분할 작업으로 확장하여 모델 아키텍처의 선택(예: padding)과 데이터 전처리 방식(예: cropping)이 어떻게 Shortcut을 유도하는지 상세히 다룬다.

## 🛠️ Methodology

### Shortcut A: 태아 초음파의 캘리퍼 및 텍스트

- **파이프라인**: 3,775장의 태아 초음파 영상(머리, 대퇴골, 복부, 자궁경부)을 사용하여 DTU-Net 아키텍처로 모델을 학습시켰다.
- **검증 방법**:
    1. 원본 이미지(텍스트/캘리퍼 포함)와 Inpainting 기법으로 해당 표식들을 제거한 'Clean' 이미지 세트 간의 성능 차이를 비교하였다.
    2. 실제 임상 영상에서 캘리퍼가 그려지는 과정 중에 분할 결과가 어떻게 변하는지 동적으로 분석하였다.
- **완화 전략**: Inpainting을 통해 표식이 제거된 이미지로 모델을 다시 학습시켜, 모델이 표식이 아닌 해부학적 특징에 집중하도록 강제하였다.

### Shortcut B: Zero-padding과 Center-cropping의 결합

- **시스템 구조**: ISIC 2017 피부 병변 데이터셋과 표준 U-Net 아키텍처를 사용하였으며, 손실 함수로는 Dice Loss를 적용하였다.
- **메커니즘 설명**:
  - **Center-cropping Bias**: 의료 데이터셋의 특성상 관심 영역(ROI)인 병변이 이미지 중앙에 위치하는 경향이 있다.
  - **Zero-padding**: CNN에서 출력 크기를 유지하기 위해 입력 경계에 0을 채우는 Zero-padding을 사용하면, 경계 근처 픽셀들의 수용장(receptive field)에는 항상 0의 띠(band)가 포함된다.
  - **상관관계 형성**: 모델은 "수용장에 0의 띠가 포함된 픽셀(경계 픽셀) $\rightarrow$ 배경"이라는 단순한 규칙을 학습하게 된다. 결과적으로 모델은 이미지 경계 근처에 위치한 실제 병변을 배경으로 오인하여 분할하지 못하게 된다.
- **완화 전략**: 훈련 시 Center-cropping 대신 Random cropping을 적용하여 병변이 이미지의 다양한 위치에 나타나도록 함으로써, 경계 픽셀과 배경 간의 상관관계를 제거하였다.

## 📊 Results

### Shortcut A 결과

- **정량적 성능**: 표식이 포함된 테스트 세트보다 표식을 제거한 'Clean' 세트에서 Dice 계수가 하락하였다 (예: 머리 부위 $76.97 \pm 5.10 \rightarrow 70.85 \pm 8.24$). 이는 모델이 표식을 Shortcut으로 사용했음을 의미한다.
- **안정성 분석**: 영상이 고정되어 있음에도 캘리퍼가 추가됨에 따라 분할 결과가 크게 요동치는 현상이 관찰되었다. 표식을 제거하고 학습한 Mitigation 모델은 훨씬 안정적인 성능을 보였다.

### Shortcut B 결과

- **경계 성능 저하**: 원본 데이터로 학습한 모델($M_{ori}$)은 병변이 이미지 중앙에서 멀어질수록 Dice Score가 급격히 하락하는 양상을 보였다.
- **완화 효과**: Random cropping으로 학습한 모델($M_{crop}$)은 병변의 위치와 상관없이 일관된 분할 성능을 유지하였으며, 이미지 경계에 걸친 병변도 정확하게 분할해냈다.
- **데이터 분석**: 주요 의료 이미지 벤치마크(ISIC, LIDC-IDRI, BraTS 등)의 마스크 분포를 분석한 결과, 상당수 데이터셋이 Center-cropping 경향을 보이고 있어 많은 기존 모델들이 이 Shortcut에 노출되었을 가능성이 높음을 시사하였다.

## 🧠 Insights & Discussion

### Shortcut Learning vs Overfitting

본 논문은 두 개념을 명확히 구분한다. Overfitting은 모델이 훈련 데이터의 무작위 노이즈까지 학습하여 일반화 능력이 떨어지는 것이지만, Shortcut Learning은 데이터 내에 실제로 존재하는 '단순하고 오해의 소지가 있는 패턴'을 학습하는 것이다. 결과적으로 둘 다 일반화 실패로 이어지지만, 그 메커니즘은 완전히 다르다.

### Domain Adaptation의 한계

Shortcut Learning이 발생한 모델에 Domain Adaptation을 적용하는 것은 효과가 없을 수 있다. Domain Adaptation은 원본 도메인에서 학습한 패턴이 타겟 도메인으로 전이 가능하다는 가정을 전제로 하지만, Shortcut Learning 모델은 처음부터 실제 패턴이 아닌 잘못된 단서를 학습했기 때문에 전이할 유의미한 패턴 자체가 존재하지 않기 때문이다.

### 비판적 해석 및 논의

- **ViT의 가능성**: Vision Transformer(ViT)는 Padding을 사용하지 않으므로 Zero-padding 기반의 Shortcut에서는 자유로울 수 있으나, Positional Encoding이 중앙 편향(center bias)을 인코딩할 가능성이 있어 여전히 연구 대상이다.
- **실무적 시사점**: 많은 피부 병변 분할 논문 중 15% 미만이 Random cropping을 사용했다는 점은, 학계가 인지하지 못한 채 상당수의 모델이 구조적 Shortcut에 의존해 성능을 부풀렸을 가능성을 제기한다.

## 📌 TL;DR

본 연구는 의료 이미지 분할 모델이 단순한 마커(캘리퍼, 텍스트)나 아키텍처 특성(Zero-padding $\times$ Center-crop)을 이용한 **Shortcut Learning**에 취약함을 입증하였다. 특히 CNN의 padding 특성이 데이터셋의 중앙 집중 경향과 결합될 때 경계 영역의 분할 실패를 유발한다는 점을 밝혀냈으며, 이를 해결하기 위해 **Inpainting**과 **Random cropping**이라는 간단하지만 효과적인 완화 전략을 제시하였다. 이는 향후 의료 AI 모델의 신뢰성과 강건성을 확보하기 위해 분할 작업에서도 Shortcut 검증이 필수적임을 시사한다.
