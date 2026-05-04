# MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation

Gurucharan Marthi Krishna Kumar, Aman Chadha, Janine Mendola, Amir Shmuel (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 정확한 진단과 치료 계획 수립에 있어 핵심적인 역할을 수행한다. 최근 Vision Transformer(ViT) 기반의 모델들이 분할 작업에서 강력한 성능을 보여주고 있으나, 이러한 모델들은 일반적으로 방대한 양의 학습 데이터를 필요로 한다. 하지만 실제 임상 환경에서는 주석이 달린 고품질의 데이터를 대량으로 확보하는 것이 매우 어려우며, 이는 데이터 부족 상황에서 모델의 성능 저하와 일반화 능력 부족으로 이어진다.

본 논문의 목표는 사전 학습된 대규모 언어 모델(Large Language Model, LLM)의 레이어를 ViT 기반 분할 모델에 통합함으로써, 데이터가 제한된 환경에서도 특징 정제(Feature Refinement) 능력을 높이고 세그멘테이션 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 사전 학습된 LLM의 트랜스포머 블록이 일종의 **'잔차 어텐션 부스터(Residual Attention Booster)'** 역할을 수행하게 하는 것이다. LLM은 방대한 텍스트 데이터를 통해 복잡한 관계적 구조와 추상적인 시맨틱 표현을 학습하였으며, 이를 시각적 특징에 적용하면 중요한 이미지 영역에 더 잘 집중할 수 있다는 '정보 필터링 가설(Information Filtering Hypothesis)'에 기반한다. 

이를 위해 저자들은 Llama-3.1-8B의 가중치를 고정(Frozen)한 상태로 ViT 인코더와 디코더 사이에 배치하고, Low-Rank Adaptation(LoRA)을 통해 효율적으로 미세 조정하는 **MedVisionLlama** 프레임워크를 제안한다.

## 📎 Related Works

기존의 의료 영상 분할은 주로 U-Net과 같은 CNN 기반 방법론이 주도해 왔으나, CNN은 국소적 특징 추출에는 능숙하지만 장거리 문맥(Long-range context)을 파악하는 데 한계가 있다. 이를 해결하기 위해 UNETR나 Swin-UNETR와 같은 하이브리드 CNN-Transformer 모델들이 등장하였다. 

최근에는 시각-언어 모델(Vision-Language Models, VLMs)을 통해 텍스트 가이드를 제공하거나 frozen transformer 블록을 통합하려는 시도가 있었으나, 대부분의 연구가 이미지 수준의 분류(Classification)나 고수준 출력에 집중되어 있었다. 픽셀 단위의 정밀한 예측이 필요한 dense prediction 작업, 특히 의료 영상 분할 분야에서 LLM의 frozen 블록을 직접적으로 통합하여 성능을 높이려는 시도는 여전히 미흡한 상태였다.

## 🛠️ Methodology

### 전체 시스템 구조
MedVisionLlama는 기본 ViT 기반 분할 네트워크에 사전 학습된 Llama 트랜스포머 블록을 통합한 구조를 가진다. 전체 파이프라인은 다음과 같은 단계로 구성된다.

1. **Base ViT Encoder ($V_E$):** 입력 이미지 $X$를 패치로 나누고 임베딩하여 잠재 표현(Latent Representation) $P$를 생성한다.
2. **Dimension Mapping Layer 1 ($V_{D1}$):** ViT 인코더의 출력 차원과 LLM 블록의 입력 차원이 서로 다르기 때문에, 이를 맞추기 위해 LoRA 기반의 학습 가능한 매핑 레이어를 사용한다.
3. **Frozen LLM Block ($V_{LLM}$):** Llama-3.1-8B의 사전 학습된 가중치를 사용하며, 학습 과정에서 가중치는 고정된다. 이 블록은 시각적 토큰을 입력받아 시맨틱 문맥과 장거리 의존성을 반영하여 특징을 정제한다.
4. **Dimension Mapping Layer 2 ($V_{D2}$):** LLM에서 처리된 특징 $Q$를 다시 ViT 디코더가 받아들일 수 있는 원래의 잠재 공간 차원으로 되돌린다.
5. **ViT Decoder ($V_D$):** 정제된 특징 $Q$를 바탕으로 최종 분할 맵 $Y$를 재구성한다.

### 주요 방정식
기본 ViT 모델의 흐름은 다음과 같다.
$$V_E(X) \to P, \quad V_D(P) \to Y$$

MedVisionLlama의 통합된 흐름은 다음과 같이 정의된다.
$$V_E(X) \to P, \quad V_{D1} V_{LLM}(P) V_{D2} \to Q, \quad V_D(Q) \to Y$$

### 학습 절차 및 세부 설정
- **학습 대상:** $V_E$, $V_D$, 그리고 두 개의 매핑 레이어 $V_{D1}, V_{D2}$는 학습 가능하며, $V_{LLM}$은 고정된 상태에서 LoRA 어댑터만을 통해 적응시킨다.
- **LoRA 적용:** 매핑 레이어와 LLM 블록 내부의 특정 레이어에 LoRA를 적용하여 파라미터 업데이트 효율을 높였다.
- **손실 함수:** 클래스 불균형이 심한 의료 영상의 특성을 고려하여 Dice Loss와 Binary Cross Entropy(BCE) Loss를 함께 사용하였다.
- **기타 설정:** LLM의 언어적 맥락에서 사용되는 Rotary Positional Embeddings와 Attention Masks는 시각적 입력과의 호환성을 위해 제거하였다.

## 📊 Results

### 실험 환경
- **데이터셋:** Medical Segmentation Decathlon (MSD) 챌린지의 10가지 작업(MRI 및 CT 모달리티 포함)을 사용하였다.
- **비교 대상:** 표준 ViT-Baseline 및 UNet++, UNETR, nnU-Net, MissFormer, TransUNet, Swin-UNet 등의 최신 SOTA 모델들과 비교하였다.
- **평가 지표:** Dice Coefficient, Jaccard Index, 95th percentile Hausdorff Distance (HD95), Specificity, Sensitivity, Normalized Surface Dice (NSD)를 사용하였다.

### 주요 결과
1. **정량적 성능 향상:** 10개 모든 작업에서 MedVisionLlama가 ViT-Baseline보다 높은 성능을 기록하였다. 특히 60번의 모든 지표 비교에서 우위를 보였으며, 그 중 41번은 통계적으로 유의미한($p < 0.05$) 향상을 나타냈다.
2. **SOTA 모델 대비 성능:** 평균 Dice score 기준, MedVisionLlama(0.87)는 Swin-UNet(0.85)이나 MissFormer(0.84)보다 높은 수치를 기록하며 가장 강력한 성능을 입증하였다.
3. **Few-shot 성능:** 학습 데이터의 10%와 30%만 사용한 환경에서도 MedVisionLlama는 Baseline보다 훨씬 빠르게 수렴하였고, 더 높은 Dice score를 달성하여 데이터 효율성을 입증하였다.
4. **정성적 분석:** Activation Map 분석 결과, Baseline은 노이즈가 많고 위치 추적이 부정확한 반면, MedVisionLlama는 해부학적으로 중요한 영역에 더 날카롭고 안정적으로 집중하는 모습을 보였다.

## 🧠 Insights & Discussion

### 성능 향상의 원인 분석
본 연구는 단순히 모델의 파라미터 수를 늘린 것이 성능 향상의 원인이 아님을 증명하기 위해, 파라미터 수를 비슷하게 맞춘 `ViT-Depth`(깊은 모델)와 `ViT-MLP`(대형 MLP 추가 모델)와 비교 실험을 진행하였다. 결과적으로 MedVisionLlama가 압도적인 성능을 보였으며, 이는 LLM의 사전 학습된 가중치가 제공하는 **시맨틱 추상화 능력과 어텐션 정제 기능**이 핵심임을 시사한다.

### 일반 LLM vs 의료 특화 LLM
BioGPT, ClinicalBERT, BioBERT와 같은 의료 특화 LLM을 적용했을 때, 일반 Llama 모델과 비교하여 통계적으로 유의미한 차이가 없었다($p > 0.05$). 이는 의료 영상 분할 작업에서의 성능 향상이 '도메인 특화 지식'보다는 LLM이 거대 말뭉치를 통해 학습한 '일반적인 특징 추출 및 관계 표현 능력'에 더 크게 의존한다는 점을 보여준다.

### LoRA의 효율성
Linear Projection 레이어를 사용하는 것보다 LoRA를 통해 내부 가중치를 미세하게 조정하는 것이 더 적은 학습 가능 파라미터로 더 높은 정확도를 달성하였다. 또한, LoRA Rank 4가 정확도와 파라미터 효율성 사이의 최적의 트레이드오프를 제공함을 확인하였다.

## 📌 TL;DR

본 논문은 사전 학습된 **Llama-3.1-8B의 frozen 레이어를 ViT 기반 의료 영상 분할 모델의 중간에 삽입**하여, 데이터가 부족한 환경에서도 특징 정제 능력을 극대화하는 **MedVisionLlama**를 제안한다. 

- **핵심 기여:** LLM을 '잔차 어텐션 부스터'로 활용하여 10가지 MSD 작업에서 SOTA 수준의 성능을 달성하였으며, 특히 Few-shot 상황에서 탁월한 일반화 능력을 보였다.
- **의의:** 의료 특화 모델이 아니더라도 거대 언어 모델의 구조적 사전 지식이 시각적 작업(Segmentation)에 전이될 수 있음을 입증하였으며, 이는 향후 제한된 의료 데이터를 극복하기 위한 효율적인 모델 설계 방향을 제시한다.