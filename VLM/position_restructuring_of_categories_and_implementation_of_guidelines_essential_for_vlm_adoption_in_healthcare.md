# Restructuring of Categories and Implementation of Guidelines Essential for VLM Adoption in Healthcare

Amara Tariq, Rimita Lahiri, Charles Kahn, Imon Banerjee (2025)

## 🧩 Problem to Solve

본 논문은 의료 분야에서 Vision-Language Model(VLM)의 도입을 가속화하기 위해, 연구 보고 및 평가의 표준화된 가이드라인이 부재하다는 문제를 제기한다. VLM의 개발 및 적용 과정은 단순히 모델을 학습시키는 것을 넘어, 사전 학습(Pretraining), 도메인 정렬(Domain Alignment), 특정 작업 최적화(Finetuning) 등 다단계의 복잡한 과정을 거친다.

기존의 머신러닝 보고 표준인 TRIPOD+AI나 CLAIM과 같은 체크리스트는 단일 모달리티 기반의 전통적인 모델 설계에 최적화되어 있어, VLM의 다단계 학습 과정이나 Zero-shot/Few-shot Prompting과 같은 최신 방법론을 포괄하지 못한다. 이러한 표준의 부재는 연구자들의 선택적 보고(Selective reporting)로 이어지며, 결과적으로 의료 AI 시스템의 핵심인 재현성(Reproducibility)과 신뢰성을 저하시키는 심각한 문제를 야기한다. 따라서 본 논문의 목표는 VLM 연구의 특성을 반영한 새로운 분류 체계를 제안하고, 이에 따른 상세한 보고 표준과 체크리스트를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 VLM 연구를 체계적으로 분류하고, 각 단계에 맞는 엄격한 보고 가이드라인을 제시하여 의료 VLM 연구의 투명성과 재현성을 확보하는 것이다. 주요 기여 사항은 다음과 같다.

1. **VLM 연구 분류 체계 제안**: VLM 연구를 개발 및 적용 전략에 따라 네 가지 범주(VLM Pretraining, Domain-specific Finetuning, Task-specific Finetuning, Prompting-based VLM studies)로 구조화하였다.
2. **상세 보고 표준 수립**: 모델 설계, 학습 절차, 데이터셋 구성, 성능 평가의 각 단계에서 반드시 보고해야 할 핵심 항목들을 정의하였다. 특히 사전 학습 목적 함수(Objective function)와 데이터 누수(Data leakage) 방지를 위한 엄격한 기준을 제시하였다.
3. **표준 체크리스트 제공**: 저자와 리뷰어가 논문의 품질과 재현 가능성을 객관적으로 평가할 수 있도록 통합된 체크리스트를 구현하였다.

## 📎 Related Works

논문에서는 기존의 다중 모달리티 학습 방식과 최신 VLM의 차이점을 다음과 같이 설명한다.

- **전통적인 데이터 퓨전(Multi-modal data fusion)**: 과거에는 CNN(이미지)과 RNN/Transformer(텍스트)를 통해 각각 임베딩을 추출한 후, 단순 결합(Concatenation)이나 투영(Projection)을 통해 퓨전하는 Late, Early, Joint fusion 방식을 사용하였다. 그러나 이러한 방식은 두 모달리티 간의 복잡한 상호의존성을 모델링하는 능력이 부족하며, 단순한 상관관계만을 학습하는 경향이 있다.
- **최신 VLM (예: CLIP)**: 최신 VLM은 단순한 정적 퓨전이 아니라, 학습 전 과정에서 시각과 언어 정보를 지속적으로 정렬(Alignment)하고 통합하는 동적 프로세스를 따른다. 이를 통해 교차 모달리티 추론(Cross-modal reasoning)이 가능해졌으며, 의료 분야에서도 흉부 X-ray 진단, 종양 검출 등에 광범위하게 적용되고 있다.
- **기존 체크리스트의 한계**: TRIPOD+AI 및 CLAIM은 전통적인 ML 모델의 실험 절차에는 유용하지만, VLM의 다단계 학습 과정(Pretraining $\rightarrow$ Domain alignment $\rightarrow$ Finetuning)과 다양한 채택 방식(Zero-shot, Few-shot)을 반영하지 못해 VLM 연구의 특수성을 충분히 담아내지 못한다.

## 🛠️ Methodology

본 논문은 VLM 연구를 네 가지 카테고리로 분류하고, 각 단계별 보고 가이드라인을 제안하는 방법론을 취한다.

### 1. VLM 연구 분류 체계 (Categorization Scheme)

VLM 연구를 다음의 네 가지 단계로 구분한다.

- **VLM Pretraining**: 이미지와 텍스트의 관계를 모델링하여 공유 표현 공간(Shared representation space)을 학습하는 단계이다.
- **Domain-specific Finetuning**: 일반 도메인에서 학습된 VLM을 의료와 같은 특정 도메인에 적응시키는 단계로, 여러 하위 작업(Downstream tasks)에 적용 가능한 일반적 능력을 배양한다.
- **Task-specific Finetuning**: 특정 진단이나 예측 등 매우 구체적인 의료 작업의 성능을 높이기 위해 가중치를 미세 조정하는 단계이다.
- **Prompting-based VLM studies**: 모델의 가중치를 수정하지 않고, 텍스트나 이미지 프롬프트를 통해 모델의 기존 지식을 인출하여 작업을 수행하는 방식이다.

### 2. 모델 설계 및 학습 보고 표준

학습 목적 함수(Training Objective)를 다음과 같이 분류하여 보고할 것을 권고한다.

- **Masked prediction**: 마스킹된 영역을 복원하는 방식 (예: MLM, MIM).
- **Contrastive learning**: 쌍을 이루는 이미지-텍스트의 유사도는 높이고, 무관한 쌍의 유사도는 낮추는 방식.
- **Image-text matching**: 두 모달리티가 서로 일치하는지 여부를 예측하는 방식.
- **Hybrid strategies**: 위 방법론들을 조합하여 사용하는 방식.

또한, 다중 목적 함수를 사용할 경우 각 함수가 성능에 기여하는 바를 파악하기 위한 **Ablation Study**를 필수적으로 수행하고 보고해야 한다.

### 3. 데이터셋 보고 표준

데이터셋을 다음 다섯 가지로 엄격히 구분하여 보고할 것을 제안한다.
$$\text{Dataset Split} = \{ \text{Pretraining, Domain-FT, Task-FT, Domain-test, Task-test} \}$$
특히, 사전 학습 데이터와 도메인 테스트 데이터 간의 중복을 방지하여 **데이터 누수(Data leakage)** 문제를 원천적으로 차단해야 함을 강조한다. 또한 의료 데이터의 특성상 환자 개인정보 보호(HIPAA, GDPR 준수)와 데이터 다양성(인종, 성별, 장비 등)에 대한 상세 기술을 요구한다.

### 4. 성능 평가 및 편향 분석 표준

- **평가 지표**: 작업 성격에 따라 Recall@K, AUROC, F-score, BLEU, Dice coefficient 등을 구분하여 사용한다.
- **벤치마킹 전략**: Pretraining 연구는 이미지$\rightarrow$텍스트 및 텍스트$\rightarrow$이미지 양방향 검색(Retrieval) 성능을 모두 보고해야 하며, Domain-FT 연구는 학습된 표현의 전이 가능성(Transferability)을 입증해야 한다.
- **편향 분석(Bias Analysis)**: 시각 데이터, 언어 데이터, 그리고 정렬 과정에서 발생하는 편향을 측정해야 하며, 통계적 패리티(Statistical parity), 불균등 영향(Disparate impact), 균등 기회(Equalized odds) 등의 지표를 통해 집단 간/개인 간 공정성을 평가해야 한다.

## 📊 Results

본 논문은 새로운 모델을 제안하고 실험하는 연구 논문이 아니라, 연구 방법론의 표준을 제안하는 **Position Paper**이다. 따라서 전통적인 의미의 정량적 실험 결과는 제시되지 않는다. 대신, 본 논문이 도출한 결과물은 다음과 같다.

1. **VLM 연구 분류 프레임워크**: VLM 개발 단계에 따른 4분류 체계 확립.
2. **보고 가이드라인**: 모델 설계, 데이터셋, 성능 평가의 각 섹션에서 다루어야 할 필수 항목 정의.
3. **표준 체크리스트 (Appendix Table 2)**: 논문 제목부터 결론, 공공 액세스까지 이어지는 상세 검토 항목 리스트를 작성하여 실제 리뷰 프로세스에 적용 가능하도록 구현하였다.

## 🧠 Insights & Discussion

본 논문은 VLM이 의료 현장에 실제로 도입되기 위해서는 단순히 성능 수치를 높이는 것보다, 그 성능이 어떻게 도출되었는지에 대한 **투명한 보고와 재현 가능성**이 더 중요하다는 통찰을 제공한다.

**강점**:

- VLM의 복잡한 다단계 학습 파이프라인을 체계적으로 구조화하여, 연구자들이 간과하기 쉬운 데이터 누수나 편향 문제를 명시적으로 지적하였다.
- 이론적인 제안에 그치지 않고, 실제 논문 심사 과정에서 사용할 수 있는 체크리스트를 제공함으로써 실무적인 활용도를 높였다.

**한계 및 논의사항**:

- 본 가이드라인은 2020년에서 2024년 사이의 문헌을 바탕으로 작성되었으므로, 향후 VLM의 학습 패러다임이 급격히 변할 경우(예: 완전히 새로운 아키텍처의 등장) 수정이 필요할 수 있다.
- 의료 데이터의 개인정보 보호 문제로 인해 모델 가중치를 공개하지 못하는 현실적인 제약이 재현성을 저해하는 큰 걸림돌이 됨을 언급하며, 이에 대한 제도적 고민이 필요함을 시사한다.

## 📌 TL;DR

본 논문은 의료 분야 VLM 연구의 재현성과 신뢰성을 높이기 위해 **연구 분류 체계 $\rightarrow$ 보고 표준 $\rightarrow$ 검토 체크리스트**로 이어지는 표준화 프레임워크를 제안한다. VLM의 다단계 학습 특성을 반영하여 데이터 누수를 방지하고, 다각도의 편향 분석과 투명한 성능 보고를 강제함으로써, 의료 AI가 실험실을 넘어 실제 임상 현장에 안전하게 적용될 수 있는 기반을 마련하고자 한다.
