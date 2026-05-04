# On generalisability of segment anything model for nuclear instance segmentation in histology images

Kesi Xu, Lea Goetz, Nasir Rajpoot (2023)

## 🧩 Problem to Solve

본 연구는 디지털 병리 이미지(Digital histology images)에서 핵심적인 분석 단계인 핵 인스턴스 분할(Nuclear instance segmentation)의 일반화 성능(Generalisability) 문제를 해결하고자 한다. 핵 분할 결과는 암 등급 판정(Cancer grading), 종양 미세환경 분석(Tumour microenvironment analysis), 생존 분석(Survival analysis) 등 하위 분석(Downstream analysis)에 필수적으로 사용된다.

그러나 기존의 머신러닝(ML) 기반 분할 모델들은 훈련 데이터와 다른 도메인의 데이터셋에 적용했을 때 성능이 급격히 저하되는 일반화 부족 문제를 겪고 있다. 임상 적용을 위해서는 서로 다른 센터나 장비에서 생성된 이미지에서도 강건하게 작동하는 모델이 필수적이다. 따라서 본 논문의 목표는 최근 공개된 기초 모델(Foundation model)인 SAM(Segment Anything Model)이 핵 인스턴스 분할 작업에서 어느 정도의 일반화 능력을 갖추고 있는지 평가하고, 이를 자동화할 수 있는 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM의 프롬프트 기반 분할 능력을 활용하여 병리 이미지의 핵 분할을 자동화하고, 그 일반화 성능을 검증하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **자동화 파이프라인 제안**: SAM은 기본적으로 사용자 상호작용(프롬프트)이 필요하므로, 이를 자동화하기 위해 핵 검출(Nucleus detection) 모델을 전단계에 배치하여 자동으로 시각적 프롬프트(Visual prompts)를 생성하는 2단계 구조를 제안하였다.
2.  **효율적인 미세 조정(Fine-tuning) 전략**: SAM의 거대한 파라미터를 모두 학습시키는 대신, Image encoder와 Prompt encoder는 동결(Freeze)하고 Mask decoder만을 미세 조정하여 병리 이미지 특성에 최적화하였다.
3.  **도메인 일반화 검증**: 대규모 데이터셋인 Lizard 데이터셋을 활용하여, 학습에 사용되지 않은 외부 데이터셋(TCGA)에 대한 제로샷(Zero-shot) 및 미세 조정 모델의 일반화 성능을 정량적으로 분석하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구 및 접근 방식을 언급한다.

-   **상호작용형 분할(Interactive Segmentation)**: NuClick과 같은 방법론은 사용자가 핵 내부에 클릭을 수행하여 마스크를 생성하는 방식을 취한다. SAM 역시 유사한 메커니즘을 가지나, 훨씬 더 방대한 데이터로 사전 학습된 기초 모델이라는 점에서 차이가 있다.
-   **지도 학습 기반 분할(Supervised Learning)**: HoVer-Net과 같은 SOTA(State-of-the-art) 모델들은 지도 학습을 통해 핵을 분할하고 분류한다. 이러한 모델들은 특정 데이터셋에서는 높은 성능을 보이지만, 도메인 변화에 취약한 한계가 있다.
-   **차별점**: 본 연구는 특정 도메인에 과적합된 지도 학습 모델과 달리, 거대 데이터셋으로 학습된 SAM의 제로샷 성능과 최소한의 튜닝만으로 달성되는 일반화 능력을 비교 분석함으로써 기초 모델의 가능성을 탐색했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
본 논문에서 제안하는 자동 핵 인스턴스 분할 파이프라인은 다음과 같은 2단계(Two-stage) 구조로 구성된다.

**1단계: 핵 검출(Nucleus Detection)**
-   입력 이미지로부터 핵의 위치를 빠르게 찾아내기 위해 미세 조정된 $\text{YOLOv8}$ 모델을 사용한다.
-   이 단계의 결과물로 각 핵을 감싸는 바운딩 박스(Bounding boxes)가 생성된다.

**2단계: 핵 분할(Nuclear Segmentation with SAM)**
-   1단계에서 검출된 바운딩 박스의 중심점(Centre points)을 SAM의 시각적 프롬프트로 사용한다.
-   SAM의 구조는 크게 $\text{Image Encoder}$, $\text{Prompt Encoder}$, 그리고 $\text{Mask Decoder}$로 나뉜다.
-   $\text{Image Encoder}$가 이미지 특징을 추출하고, $\text{Prompt Encoder}$가 중심점 좌표를 임베딩하여 두 정보가 $\text{Mask Decoder}$에서 병합되어 최종 인스턴스 마스크를 생성한다.

### 학습 절차 및 설정
-   **미세 조정(Fine-tuning)**: SAM의 모든 가중치를 학습시키지 않고, $\text{Image Encoder}$와 $\text{Prompt Encoder}$는 고정한 상태에서 $\text{Mask Decoder}$만을 Lizard 학습 데이터셋을 사용하여 미세 조정하였다. 이때 프롬프트로는 핵의 중심점을 사용하였다.
-   **추론 절차**: $\text{YOLOv8}(\text{Input Image}) \rightarrow \text{Bounding Boxes} \rightarrow \text{Center Points} \rightarrow \text{SAM}(\text{Image, Center Points}) \rightarrow \text{Final Mask}$ 순으로 진행된다.

## 📊 Results

### 실험 설정
-   **데이터셋**: Lizard 데이터셋을 사용하였으며, 이는 6개 센터(GlaS, CRAG, CoNSeP, DigestPath, PanNuke, TCGA)의 데이터를 포함한다.
-   **검증 방법**: 5개 센터의 데이터를 학습에 사용하고, 완전히 독립된 외부 데이터셋인 $\text{TCGA}$ 데이터를 테스트셋으로 사용하여 도메인 일반화 성능을 평가하였다.
-   **평가 지표**: 
    -   $\text{Dice score}$: 핵과 배경 간의 시맨틱 분할 성능을 측정한다.
    -   $\text{Panoptic Quality (PQ)}$: 인스턴스 분할 성능을 측정하며, 다음과 같이 정의된다.
    $$PQ = DQ \times SQ$$
    (여기서 $DQ$는 Detection Quality, $SQ$는 Segmentation Quality를 의미한다.)

### 정량적 결과
1.  **상호작용형 방법론 비교 (Table 1)**:
    -   정답(Ground truth) 중심점 프롬프트를 사용했을 때, 미세 조정된 SAM($PQ=0.678$)이 NuClick($PQ=0.663$)보다 우수한 성능을 보였다.
    -   바운딩 박스 프롬프트를 사용한 SAM이 가장 높은 성능($PQ=0.703$)을 기록했으나, 임상 현장에서 모든 핵에 바운딩 박스를 그리는 것은 비현실적이라고 분석하였다.

2.  **도메인 일반화 성능 비교 (Table 2 - TCGA 테스트)**:
    -   학습되지 않은 TCGA 데이터셋에 대해 $\text{YOLOv8} + \text{Finetuned SAM}$ 조합은 $PQ=0.569$를 기록하였다.
    -   이는 기존의 강력한 모델인 HoVer-Net($PQ=0.514$)보다 $3.3\%$ 높은 수치이며, U-Net이나 Micro-Net보다 월등히 높은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 가능성
본 연구는 SAM이 병리 이미지라는 특수한 도메인에서도 매우 강력한 일반화 능력을 갖추고 있음을 입증하였다. 특히 Mask decoder만을 미세 조정했음에도 불구하고, 완전히 새로운 도메인의 데이터(TCGA)에서 기존 SOTA 모델인 HoVer-Net을 능가했다는 점은 SAM이 Computational Pathology(CPath) 분야의 기초 모델(Foundation model)로서 가능성이 매우 높음을 시사한다.

### 한계 및 비판적 해석
-   **프롬프트 의존성**: SAM의 성능이 프롬프트의 정밀도(바운딩 박스 vs 중심점)에 따라 크게 좌우된다. 자동화를 위해 도입한 $\text{YOLOv8}$의 검출 성능이 낮을 경우, 전체 파이프라인의 성능이 함께 저하되는 병목 현상이 발생할 수 있다.
-   **계산 비용**: SAM은 거대한 비전 트랜스포머(ViT) 기반 모델이므로, 실시간 추론이나 대용량 슬라이드 이미지(Whole Slide Images) 전체에 적용하기에는 계산 자원 소모가 매우 클 것으로 예상된다. 이에 대한 구체적인 추론 시간 분석이 제시되지 않은 점은 아쉽다.
-   **가정**: 본 연구는 중심점 프롬프트만으로 충분한 성능을 낼 수 있다고 가정하였으나, 실제 복잡하게 겹쳐진 핵(Overlapping nuclei) 환경에서 중심점 하나만으로 정교한 경계를 구분해낼 수 있는지에 대한 정성적 분석이 더 보완될 필요가 있다.

## 📌 TL;DR

본 논문은 거대 기초 모델인 SAM을 병리 이미지의 핵 인스턴스 분할에 적용하고 그 일반화 성능을 평가하였다. $\text{YOLOv8}$을 이용한 핵 검출과 SAM의 마스크 디코더 미세 조정을 결합한 2단계 파이프라인을 제안하였으며, 실험 결과 학습되지 않은 외부 데이터셋(TCGA)에서도 HoVer-Net 등 기존 모델보다 뛰어난 일반화 성능을 보였다. 이는 SAM이 향후 디지털 병리 분석의 범용적인 기초 모델로 활용될 수 있음을 보여준다.