# Machine Learning and Deep Learning Methods for Building Intelligent Systems in Medicine and Drug Discovery: A Comprehensive Survey

Jignesh Chowdary G, Suganya G, Premalatha M, Asnath Victy Phamila Y, Karunamurthy K (2021)

## 🧩 Problem to Solve

본 논문은 현대 의료 진단 과정에서 발생하는 복잡성과 시간 소모, 그리고 인간 의사의 주관적 판단으로 인한 편향(bias) 및 오진 가능성이라는 문제를 해결하고자 한다. 의료 데이터는 매우 복잡한 관계를 가지고 있어 단순한 분석으로는 조기 진단이 어렵으며, 특히 유방 촬영술(mammography)과 같은 사례에서 높은 위양성(false-positive) 비율이 불필요한 추가 검사와 환자의 심리적 고통을 초래한다는 점을 지적한다. 따라서 본 연구의 목표는 16가지의 다양한 의료 전문 분야와 약물 발견(drug discovery) 영역에서 Machine Learning(ML) 및 Deep Learning(DL)이 어떻게 적용되고 있는지 종합적으로 분석하고, 이러한 지능형 시스템이 임상 의사들에게 제공할 수 있는 보조적 진단 방법으로서의 가능성을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단일 질환이 아닌, 의료 전반을 아우르는 광범위한 서베이를 수행했다는 점이다. 구체적인 기여 사항은 다음과 같다.

- **광범위한 전문 분야 분석**: 치과, 혈액학, 외과, 심장학, 폐학, 정형외과, 방사선과, 종양학, 일반의학, 정신의학, 내분비학, 신경학, 피부과, 간학, 신장학, 안과 등 16개 의료 분야와 약물 발견 및 COVID-19 진단까지 포함하는 포괄적인 리뷰를 제공한다.
- **알고리즘 및 프레임워크의 체계적 정리**: 각 분야에서 사용된 ML(SVM, Random Forest, KNN 등) 및 DL(CNN, RNN, GAN, Transfer Learning 등) 모델의 적용 사례와 그 성능을 정량적 수치와 함께 정리하여 제시한다.
- **의료 전문가에 대한 영향 분석**: AI 시스템이 의사를 대체하는 것이 아니라, 진단의 정확성을 높이고 워크플로우를 최적화하는 보조 도구(Decision Support System)로서 작용할 것임을 논의한다.
- **공공 데이터셋 리스트 제공**: 향후 연구자들이 활용할 수 있도록 다양한 의료 영상 및 수치 데이터셋의 링크를 제공하여 연구 생태계에 기여한다.

## 📎 Related Works

본 논문은 기존의 AI, ML, DL의 관계를 정의하며 시작한다. AI는 가장 넓은 개념이며, ML은 데이터로부터 예측과 분류를 수행하는 통계적 모델의 집합이고, DL은 ML의 하위 집합으로서 다층 신경망을 통해 고수준의 특징(high-level features)을 추출하는 기술이다.

기존 연구들은 주로 특정 질병(예: 암 또는 심장병) 하나에 집중하여 모델을 제안하는 경향이 있었으나, 본 논문은 이러한 개별 연구들을 통합하여 의료 시스템 전체의 관점에서 분석한다. 특히 CNN은 컴퓨터 비전 및 이미지 분류에, RNN은 순차적 데이터 분석 및 오디오 인식에, GAN은 의료 이미지 합성 및 데이터 증강(augmentation)을 통한 과적합(overfitting) 방지에 사용된다는 점을 명시하며 기존 기술들의 역할을 정의한다.

## 🛠️ Methodology

본 논문은 새로운 알고리즘을 제안하는 연구가 아니라, 기존 문헌을 분석하는 Survey Paper이다. 따라서 방법론은 문헌 선정 과정과 분석 체계로 구성된다.

### 1. 서베이 방법론 (Survey Methodology)

연구진은 IEEE, Elsevier, ACM, Springer와 같은 고품질 학술지 및 컨퍼런스에서 발행된 논문을 대상으로 하였다. 검색 키워드는 'machine learning', 'deep learning'과 더불어 16개 의료 전문 분야 명칭 및 'drug discovery'를 조합하여 사용하였으며, 경험적 연구(empirical articles)와 리뷰 논문을 모두 포함하였다.

### 2. 분석 대상 모델의 분류

논문은 적용된 기술을 크게 두 가지 범주로 나누어 분석한다.

- **Machine Learning**: 지도 학습(Supervised Learning)의 Random Forest, Decision Tree, Logistic Regression, KNN, SVM과 비지도 학습(Unsupervised Learning)의 PCA, Latent Dirichlet Analysis 등을 다룬다.
- **Deep Learning**: CNN, RNN, GAN 및 이들의 변형 구조를 다루며, 특히 사전 학습된 모델을 활용하는 Transfer Learning의 효율성을 강조한다.

### 3. 주요 분석 항목

각 의료 분야별로 다음 요소들을 중점적으로 살펴본다.

- **입력 데이터**: X-ray, CT, MRI, ECG, 혈액 샘플, 사회관계망서비스(SNS) 데이터 등.
- **사용된 모델**: 특정 질환 진단에 사용된 구체적인 아키텍처.
- **평가 지표**: Accuracy, Sensitivity, Specificity, F1-score, Dice coefficient, AUC-ROC 등.

본 논문 내에 특정 수식이나 독자적인 손실 함수는 명시되어 있지 않으며, 인용된 개별 논문들의 결과치를 테이블 형태로 요약하여 비교하는 방식을 취한다.

## 📊 Results

분석 결과, 의료 분야 전반에서 ML과 DL 모델이 인간 전문가 수준 혹은 그 이상의 성능을 보이는 사례가 다수 발견되었다.

### 1. 이미지 기반 진단 (Radiology, Oncology, Ophthalmology 등)

- **CNN의 우세**: 대부분의 영상 진단에서 CNN 기반 모델이 SVM이나 Random Forest보다 뛰어난 성능을 보였다. 특히 COVID-19 진단에서 ResNet50, VGG19 등의 전이 학습 모델이 90% 이상의 높은 정확도를 기록하였다.
- **세분화(Segmentation)**: U-Net 아키텍처가 폐 영역 분할 및 종양 검출에서 효과적으로 사용됨을 확인하였다.

### 2. 수치 및 신호 기반 진단 (Cardiology, Endocrinology, Psychiatry 등)

- **ML 모델의 효율성**: 정형 데이터(tabular data) 기반의 진단(예: 당뇨병, 갑상선 질환)에서는 SVM, Random Forest, Naive Bayes 등이 여전히 강력한 성능을 발휘한다.
- **신호 분석**: ECG 신호를 이용한 심근경색 진단에 CNN이 사용되어 노이즈가 있는 데이터에서도 높은 민감도(Sensitivity)를 보였다.

### 3. 기타 특수 분야

- **약물 발견**: SVM이 약물 설계 및 독성 예측에서 높은 성능을 보였으며, Bayesian Classifier가 약물 유사성(drug likeliness) 예측에 효과적임이 나타났다.
- **정신 의학**: SNS 데이터나 뇌파(brain waves)를 이용한 우울증 및 스트레스 진단에 MLP와 RNN 계열 모델이 적용되고 있다.

## 🧠 Insights & Discussion

### 1. AI와 의료 전문가의 관계

논문은 AI가 의사를 완전히 대체할 수 없다는 점을 분명히 한다. 그 이유는 다음과 같다.

- **신뢰와 상호작용**: 환자와의 정서적 교감 및 신뢰 구축은 기계가 수행할 수 없는 영역이다.
- **희귀 사례(Novel Cases)의 한계**: AI는 학습 데이터에 의존하므로, 전례 없는 신종 질병이나 복합적인 약물 부작용 사례에서는 성능이 급격히 저하된다.
- **보조적 역할**: 따라서 AI는 의사의 판단을 돕는 'Decision Support System'으로서, 특히 방사선과 전문의의 판독 효율을 높이고 위양성 결과를 줄이는 데 기여할 수 있다.

### 2. 한계 및 비판적 해석

본 논문은 매우 방대한 양의 연구를 요약하고 있으나, 각 연구마다 사용한 데이터셋의 규모와 전처리 방법이 상이함에도 불구하고 단순 수치(Accuracy 등)로 비교했다는 점이 한계로 보인다. 또한, 실제 임상 환경에서의 적용 가능성(Clinical Validation)보다는 모델의 실험적 성능에 치중하여 서술되어 있다.

## 📌 TL;DR

본 논문은 16개 의료 전문 분야와 약물 발견 영역에서 Machine Learning 및 Deep Learning의 적용 현황을 분석한 포괄적인 서베이 보고서이다. 이미지 데이터에는 CNN과 Transfer Learning이, 수치 데이터에는 SVM과 Random Forest가 주로 사용되며, 전반적으로 인간의 진단 편향을 줄이고 조기 진단 가능성을 높이는 결과가 확인되었다. 결론적으로 AI는 의사를 대체하는 것이 아니라, 진단의 정확성과 효율성을 극대화하는 강력한 보조 도구로서의 가치를 지닌다.
