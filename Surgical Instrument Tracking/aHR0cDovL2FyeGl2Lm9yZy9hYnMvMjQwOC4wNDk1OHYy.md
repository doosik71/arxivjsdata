# Surgical-VQLA++: Adversarial Contrastive Learning for Calibrated Robust Visual Question-Localized Answering in Robotic Surgery

Long Bai, Guankun Wang, Mobarakol Islam, Lalithkumar Seenivasan, An Wang, and Hongliang Ren (2024)

## 🧩 Problem to Solve

본 논문은 로봇 수술 환경에서 수술 이미지와 관련된 질문에 답하고, 그 답의 근거가 되는 영역을 시각적으로 제시하는 Visual Question Localized-Answering (VQLA) 문제를 해결하고자 한다.

기존의 Surgical VQA(Visual Question Answering) 모델들은 "무엇(What)"에 해당하는 정답을 생성하는 데에만 집중하여, 정답이 도출된 "어디(Where)"에 대한 시각적 정보가 결여되어 있었다. 이는 수술 교육생들이 단순히 텍스트 기반의 정답만으로는 수술 장면의 복잡한 인과관계나 "왜(Why)" 그러한 결과가 나왔는지를 추론하는 데 한계가 있다는 점을 시사한다.

또한, 실제 임상 환경의 의료 데이터는 이미지 획득 및 전송 과정에서 다양한 노이즈, 아티팩트, 혹은 이미지 품질 저하(Corruption)가 발생할 가능성이 높다. 기존 모델들은 이러한 이미지 손상에 매우 민감하여 성능이 급격히 저하되는 경향이 있으며, 이는 진단 오류나 교육적 혼란을 야기할 수 있는 심각한 안전성 문제로 이어진다. 따라서 본 논문의 목표는 정답 생성과 영역 국지화(Localization)를 동시에 수행하면서도, 다양한 이미지 손상에 대해 강건함(Robustness)을 유지하는 VQLA 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 멀티모달 특징의 정밀한 정렬(Alignment)과 보정(Calibration), 그리고 적대적 대조 학습(Adversarial Contrastive Learning)을 통해 모델의 강건성과 정확도를 동시에 높이는 것이다. 주요 기여 사항은 다음과 같다.

1. **Surgical-VQLA++ 프레임워크 제안**: 정답 생성과 국지화 사이에 인스턴스 수준의 연결을 구축하여, 글로벌한 장면 이해와 빠른 추론 속도(150.6 FPS)를 동시에 달성하였다.
2. **$\text{C}^2\text{G-ViL}$ Embedding 모듈 설계**: Calibrated Co-Attention Gated Vision-Language ($\text{C}^2\text{G-ViL}$) 모듈을 통해 시각 및 텍스트 표현의 정렬과 상호작용을 최적화하고, 글로벌 컨텍스트 보정을 통해 데이터 변동성에 대응하였다.
3. **적대적 대조 학습(Adversarial Contrastive Training) 도입**: DeiT 백본 네트워크에 적대적 샘플 기반의 대조 학습 전략을 적용하여, 미세한 섭동(Perturbation)에도 흔들리지 않는 강건한 특징 표현을 학습하도록 하였다.
4. **데이터셋 확장**: 기존 EndoVis-18 및 EndoVis-17-VQLA 데이터셋에 수술 도구 관련 쿼리를 추가하여 총 17,269개의 QA 쌍을 확보함으로써 연구의 범위를 확장하였다.

## 📎 Related Works

기존의 컴퓨터 비전 분야에서는 VQA와 Visual Grounding을 결합하여 모델의 해석 가능성을 높이려는 연구가 활발히 진행되어 왔다. 특히 의료 분야에서는 복잡한 해부학적 구조와 도메인 특화 용어로 인해 일반적인 VQA 모델을 그대로 적용하기 어려웠다.

이전의 Surgical-VQA 연구들은 주로 텍스트 정답 생성에 치중하였으며, 최근 제안된 Surgical-VQLA는 국지화 기능을 추가하였으나 객체 제안(Object Proposal) 방식의 특성상 추론 속도가 느리고 글로벌한 문맥 이해가 부족하다는 한계가 있었다. 또한, 의료 영상의 고유한 특성인 노이즈나 도메인 갭(Domain Gap)으로 인한 성능 저하 문제를 효과적으로 해결한 연구가 부족했다. 본 논문은 이러한 한계를 극복하기 위해 엔드-투-엔드(End-to-End) 구조의 글로벌 특징 추출기와 적대적 학습 기반의 강건성 강화 전략을 채택하여 기존 접근 방식과 차별화한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

Surgical-VQLA++는 시각적 특징 추출기(ResNet18), 텍스트 토크나이저, $\text{C}^2\text{G-ViL}$ 임베딩 모듈, DeiT 백본 네트워크, 그리고 정답 분류(Classification) 및 바운딩 박스 회귀(Localization)를 위한 두 개의 예측 헤드로 구성된다.

### 2. $\text{C}^2\text{G-ViL}$ 임베딩 모듈

이 모듈은 시각 및 텍스트 정보의 효과적인 융합을 위해 네 가지 단계로 작동한다.

* **Co-Attention Cross-Model Interaction**: Self-Attention과 Guided-Attention을 결합하여 두 모달리티 간의 상호작용을 극대화한다. 특히 텍스트 정보를 통해 시각 정보의 핵심 영역에 집중하도록 유도한다.
* **Multimodal Collaborated Calibration (MCC)**: 시각 임베딩을 텍스트 임베딩으로, 텍스트 임베딩을 시각 임베딩으로 각각 보정하여 두 모달리티 간의 신뢰할 수 있는 대응 관계를 구축하고 정렬한다.
* **Global Contextual Calibration (GCC)**: Pairwise Bilinear Pooling 기술을 사용하여 각 모달리티 내부의 글로벌 컨텍스트 시맨틱을 정제한다.
* **Gated Fusion**: $\tanh$ 활성화 함수와 시그모이드 게이트를 통해 시각 및 텍스트 임베딩의 융합 가중치를 동적으로 학습하여 최적의 중간 표현을 생성한다.

### 3. 적대적 대조 학습 (Adversarial Contrastive Training)

모델의 강건성을 높이기 위해 입력 임베딩에 의도적인 섭동 $\delta$를 추가한 적대적 샘플을 생성한다. 섭동은 다음과 같이 계산된다.
$$\delta = -\epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(F(x), y))$$

이후 깨끗한(Clean) 샘플의 특징 벡터 $z_i$와 섭동이 가해진(Perturbed) 샘플의 특징 벡터 $z'_j$ 사이의 코사인 유사도 $\text{Sim}(z_i, z'_j)$를 기반으로 대조 손실(Contrastive Loss) $\mathcal{L}_{CONT}$를 정의하여, 모델이 작은 변동에도 동일한 클래스의 특징을 유지하도록 학습시킨다.

### 4. 손실 함수 및 학습 절차

본 모델은 다중 작업 학습(Multi-task Learning)을 수행하며, 각 작업의 불균형을 해결하기 위해 다음과 같은 손실 함수를 사용한다.

* **QA 손실**: 클래스 불균형 문제를 해결하기 위해 Cross-Entropy 대신 Focal Loss를 사용한다.
* **Localization 손실**: 바운딩 박스 회귀를 위해 GIoU(Generalized IoU) Loss를 사용한다.
* **Uncertainty Loss**: 두 작업 간의 가중치를 수동으로 정하는 대신, 학습 가능한 파라미터 $\sigma_1, \sigma_2$를 도입하여 작업의 불확실성에 따라 가중치를 동적으로 조절한다.
$$\mathcal{L}_{VQLA} = \frac{1}{2\sigma_1^2} \mathcal{L}_{Focal} + \log\sigma_1 + \frac{1}{2\sigma_2^2} \mathcal{L}_{GIoU} + \log\sigma_2$$

최종 손실 함수 $\mathcal{L}$은 깨끗한 샘플의 VQLA 손실, 섭동 샘플의 VQLA 손실, 그리고 대조 손실의 합으로 정의된다.
$$\mathcal{L} = \mathcal{L}_{VQLA}(x; y) + \mathcal{L}_{VQLA}(x + \delta; y) + \mathcal{L}_{CONT}(x; \delta)$$

## 📊 Results

### 1. 실험 설정

* **데이터셋**: EndoVis-18-VQLA(학습 및 테스트) 및 EndoVis-17-VQLA(외부 검증)를 사용하였다.
* **비교 대상**: VisualBERT, MCAN, MUTAN, BlockTucker, CAT-ViL DeiT 및 이전 버전인 Surgical-VQLA 등이 비교 대상으로 사용되었다.
* **지표**: 정답 정확도(Accuracy), F-Score, 국지화 성능을 위한 mIoU를 측정하였다.

### 2. 주요 결과

* **정량적 성능**: Surgical-VQLA++는 모든 벤치마크 모델을 상회하는 성능을 보였다. 특히 두 번째로 성능이 좋은 모델 대비 정확도는 $1.12\%$, mIoU는 $1.55\%$ 향상되었다.
* **강건성 테스트**: 노이즈, 블러, 폐쇄(Occlusion), 디지털 손상 등 19가지 유형의 이미지 손상을 5단계 심각도로 나누어 테스트하였다. 이미지 손상 수준이 높아질수록 모든 모델의 성능이 하락하지만, Surgical-VQLA++는 타 모델 대비 훨씬 완만한 성능 하락 곡선을 그리며 높은 안정성을 보였다.
* **추론 속도**: 엔드-투-엔드 구조를 통해 150.6 FPS라는 매우 빠른 추론 속도를 달성하여 실시간 적용 가능성을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과

본 논문은 $\text{C}^2\text{G-ViL}$ 모듈과 적대적 대조 학습을 통해 의료 영상 특유의 노이즈와 도메인 갭 문제를 효과적으로 해결하였다. 특히 적대적 학습이 단순한 성능 향상을 넘어, 실제 수술 중 발생할 수 있는 카메라 렌즈의 혈흔 오염이나 조명 변화와 같은 극한 상황에서도 모델이 안정적으로 작동하게 함으로써 임상적 가치를 높였다.

### 2. 한계 및 비판적 해석

실험 결과, 조직 인식(Tissue recognition)이나 위치 파악(Location)에서는 매우 높은 성능을 보였으나, 수술 도구 식별(Instrument identification) 작업에서는 상대적으로 낮은 성능을 보였다. 이는 수술 도구들의 외형적 유사성이 매우 높기 때문으로 분석되며, 향후 더욱 세밀한 도구 특징을 추출할 수 있는 전략이 필요하다. 또한, F-Score 결과가 일부 시나리오에서 낮게 나타난 점은 정밀도(Precision)와 재현율(Recall) 간의 균형을 맞추기 위한 추가적인 융합 전략이 필요함을 시사한다.

### 3. MLLM과의 비교

최근의 거대 멀티모달 모델(MLLM)들이 뛰어난 성능을 보이지만, 본 연구의 모델은 훨씬 적은 자원으로 실시간 추론이 가능하며, 의료 분야에서 치명적일 수 있는 환각(Hallucination) 현상으로부터 비교적 자유롭다는 점에서 실용적인 우위를 가진다.

## 📌 TL;DR

본 논문은 로봇 수술 영상에서 질문에 답하고 해당 영역을 찾아내는 **Surgical-VQLA++**를 제안한다. **$\text{C}^2\text{G-ViL}$ 임베딩**을 통해 시각-언어 정보의 정렬을 최적화하고, **적대적 대조 학습**을 통해 이미지 손상에 대한 강력한 강건성을 확보하였다. 결과적으로 SOTA 모델 대비 높은 정확도와 mIoU를 달성했으며, 실시간 추론이 가능한 속도를 구현하여 실제 수술 교육 및 임상 의사결정 지원 도구로서의 높은 가능성을 보여주었다.
