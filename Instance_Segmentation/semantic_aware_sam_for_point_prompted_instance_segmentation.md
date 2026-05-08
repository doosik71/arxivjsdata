# Semantic-aware SAM for Point-Prompted Instance Segmentation

Zhaoyang Wei, Pengfei Chen, Xuehui Yu, Guorong Li, Jianbin Jiao, Zhenjun Han (2024)

## 🧩 Problem to Solve

본 논문은 매우 적은 비용의 어노테이션인 단일 지점 주석(single-point annotation)만을 사용하여 정밀한 인스턴스 분할(Instance Segmentation)을 수행하는 것을 목표로 한다.

최근 Segment Anything Model (SAM)과 같은 시각적 기반 모델(Visual Foundation Models)이 등장하여 제로샷 성능과 강력한 분할 능력을 보여주었으나, 다음과 같은 핵심적인 문제점들이 존재한다.

1. **SAM의 의미론적 모호성(Semantic Ambiguity):** SAM은 클래스 구분 능력이 없는 class-agnostic 모델이다. 따라서 특정 카테고리의 객체를 추출하려 해도, 모델이 내부적으로 높은 신뢰도를 부여하는 부분(예: '사람' 전체가 아닌 '옷' 부분)만을 분할하는 경향이 있어 정밀한 카테고리별 분할이 어렵다.
2. **MIL 기반 방식의 한계:** 다중 인스턴스 학습(Multiple Instance Learning, MIL)을 통해 제안된 마스크(proposal) 중 최적을 선택하는 방식은 두 가지 고질적인 문제를 가진다.
    - **Group Issue:** 인접한 동일 카테고리 객체들을 하나의 객체로 묶어버리는 현상이다.
    - **Local Issue:** 객체 전체가 아닌 가장 변별력이 높은(discriminative) 국소 영역만을 선택하는 현상이다.

결과적으로 본 논문은 SAM의 강력한 분할 능력에 의미론적 인지 능력을 부여하여, 포인트 프롬프트만으로도 완전 지도 학습(Fully-supervised)에 근접하는 인스턴스 분할 성능을 달성하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM이 생성한 클래스 불가지론적(class-agnostic) 마스크 제안들을 의미론적으로 필터링하고 정제하는 **SAPNet (Semantic-Aware Instance Segmentation Network)**을 설계하는 것이다.

주요 기여 사항은 다음과 같다.

- **SAPNet 프레임워크 제안:** SAM의 포인트 프롬프트 출력물에 MIL 기반의 선택 메커니즘을 결합하여 카테고리 특화된 마스크를 생성하는 엔드-투-엔드 네트워크를 제안한다.
- **Point Distance Guidance (PDG):** 동일 클래스의 포인트 간 유클리드 거리를 이용해 패널티를 부여함으로써, 인접 객체들이 하나로 묶이는 'group issue'를 해결한다.
- **Positive and Negative Proposals Generator (PNPG) 및 Box Mining Strategy (BMS):** 긍정/부정 샘플을 전략적으로 생성하고 박스 크기를 적응적으로 확장함으로써, 국소 영역만 선택되는 'local issue'를 해결한다.
- **성능 입증:** COCO 및 VOC2012 벤치마크에서 포인트 프롬프트 기반 인스턴스 분할(PPIS) 분야의 SOTA 성능을 달성하였으며, 포인트 주석과 완전 지도 학습 간의 간극을 크게 좁혔다.

## 📎 Related Works

### 1. Weakly-Supervised Instance Segmentation (WSIS)

기존 연구들은 바운딩 박스(Bounding box)나 이미지 레벨 레이블(Image-level label)과 같은 약한 지도 학습을 통해 마스크를 생성하려 했다. 하지만 바운딩 박스 기반 방법은 학습 과정이 복잡하고 객체의 전체적인 형태를 놓치는 경우가 많으며, 이미지 레벨 방법은 인스턴스 간의 분리 능력이 떨어진다는 한계가 있다.

### 2. Pointly-Supervised Detection and Segmentation (PSDS)

포인트 주석은 비용이 매우 저렴하면서도 위치 정확도가 높다. WISE-Net, BESTIE 등의 연구가 진행되었으나, 단일 포인트만으로 객체 전체의 정밀한 마스크를 복원하는 것에는 여전히 어려움이 있으며, 특히 복잡한 장면에서의 효과가 충분히 입증되지 않은 상태였다.

### 3. Prompting and Foundation Models (SAM)

SAM은 강력한 제로샷 일반화 능력을 갖추고 있어 다양한 하위 작업에 활용되고 있다. 그러나 SAM 자체는 분류(Classification) 능력이 없으므로, 이를 인스턴스 분할에 활용하기 위해서는 외부의 의미론적 정보나 추가적인 학습 메커니즘이 필수적이다.

## 🛠️ Methodology

SAPNet은 크게 두 가지 브랜치로 구성된다. 하나는 마스크 제안을 선택하고 정제하여 의사 레이블(pseudo-labels)을 생성하는 브랜치이고, 다른 하나는 생성된 의사 레이블로 학습되는 SOLOv2 기반의 인스턴스 분할 브랜치이다.

### 1. Proposal Selection Module (PSM)

SAM을 통해 생성된 마스크 제안들을 최소 경계 사각형(minimum bounding rectangle) 형태의 박스 제안으로 변환하고, 이를 MIL 방식으로 처리한다.

- **과정:** 이미지 특징 $F$와 박스 제안 $B_i$를 입력받아 RoIAlign과 FC 레이어를 거쳐 클래스 점수 $S^{cls}$와 인스턴스 점수 $S^{ins}$를 계산한다.
- **Point Distance Guidance (PDG):** 인접한 동일 클래스 객체의 중첩을 방지하기 위해 포인트 간 거리 기반 패널티 $S^{dis}$를 도입한다.
$$[W_{dis}]_{im} = \sum_{j=1, j \neq i}^{N} \|p_i - p_j\| \cdot t_{mj}$$
$$[S_{dis}]_{im} = (1/e^{-(1/[W_{dis}]_{im})})^d$$
- **PSM 손실 함수:** 최종 점수 $S$는 $S^{cls} \odot S^{ins} \odot S_{dis}$의 아다마르 곱으로 계산하며, 이진 교차 엔트로피(BCE) 손실을 사용한다.
$$L_{psm} = CE(\bar{S}, c)$$

### 2. Positive and Negative Proposals Generator (PNPG)

PSM에서 선택된 박스를 기반으로 더 정교한 학습을 위한 샘플 집합을 생성한다.

- **PPG (Positive):** 선택된 박스의 크기를 스케일 팩터 $v$와 거리 점수 $S_{dis}$를 이용하여 적응적으로 조절하여 긍정 샘플 집합 $B^+$를 확장한다.
- **NPG (Negative):** 배경 영역을 무작위 샘플링하거나, 객체 내부의 작은 부분(part)만을 포함하는 박스를 생성하여 부정 샘플 집합 $U$를 구축함으로써 'local issue'를 억제한다.

### 3. Proposals Refinement Module (PRM)

생성된 $B^+$와 $U$를 사용하여 제안된 박스를 한 번 더 정제한다.

- **손실 함수:** 긍정 샘플에는 Focal Loss를 적용하고, 배경 억제를 위해 별도의 $L_{neg}$를 도입한다.
$$L_{prm} = \alpha L_{pos} + (1-\alpha) L_{neg}$$

### 4. Box Mining Strategy (BMS)

MIL이 여전히 국소 영역을 선호하는 문제를 해결하기 위해, 선택된 박스($box_{select}$)를 주변 제안들과 적응적으로 병합하여 확장하는 전략이다. IoU와 크기 조건을 고려하여 객체의 실제 경계에 가깝게 박스를 넓히고 최종적으로 $box_{prm}$을 도출한다.

### 5. 전체 학습 절차 및 손실 함수

최종적으로 정제된 마스크 $Mask_{prm}$과 SAM의 필터링된 마스크 $Mask_{sam}$을 함께 사용하여 분할 브랜치를 지도한다. 전체 손실 함수는 다음과 같다.
$$L_{total} = L_{mask} + L_{cls} + \lambda \cdot L_{psm} + L_{prm}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** MS COCO (80 클래스), VOC2012SBD (20 클래스).
- **지표:** mAP (COCO), $AP_{25, 50, 75}$ (VOC).
- **백본:** ResNet-50 및 ResNet-101.

### 2. 주요 결과

- **COCO 데이터셋:** ResNet-50 기반으로 31.2 AP를 달성하여 기존 SOTA인 BESTIE(17.7 AP)와 AttnShift(21.2 AP)를 크게 상회한다. 특히 ResNet-101과 3x 학습 스케줄 적용 시 34.6 AP를 기록하여, 바운딩 박스 기반 방법인 BoxInst(33.2 AP)보다 높은 성능을 보였다.
- **VOC2012 데이터셋:** $AP_{50}$ 기준 64.8 AP를 달성하였으며, 이는 완전 지도 학습 모델인 Mask R-CNN 성능의 약 92.3%에 도달한 수치이다.
- **분석:** SAM의 top-1 마스크만 사용했을 때(24.6 AP)보다 SAPNet의 정제 과정을 거쳤을 때(31.2 AP) 성능이 대폭 향상됨을 확인하였다.

### 3. Ablation Study

- **구성 요소 효과:** MIL $\rightarrow$ PDG $\rightarrow$ PNPG $\rightarrow$ BMS $\rightarrow$ MPS 순으로 적용할 때 mAP가 단계적으로 상승(26.8 $\rightarrow$ 27.5 $\rightarrow$ 29.7 $\rightarrow$ 30.8 $\rightarrow$ 31.2)함을 보였다.
- **PNPG 분석:** 긍정 샘플(PPG)보다 부정 샘플(NPG)을 통한 배경 억제 및 부분 영역 제거가 성능 향상에 더 큰 기여를 함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 논문은 SAM이라는 강력한 도구를 사용하면서도, 그 한계점인 '의미론적 모호성'을 MIL과 포인트 거리 가이드, 그리고 정교한 샘플링 전략으로 해결하였다. 특히 포인트 주석이라는 극도로 적은 정보만으로 바운딩 박스 기반의 지도 학습 성능을 추월했다는 점은 매우 고무적이다.

### 한계 및 논의사항

- **학습 비용:** 추론(Inference) 시에는 분할 브랜치만 사용하므로 비용이 들지 않지만, 학습 단계에서는 SAM의 마스크 생성 및 다단계 정제 과정(PSM $\rightarrow$ PRM)이 포함되어 연산 오버헤드가 발생한다.
- **가정:** 본 모델은 포인트 주석이 정확하게 객체 내부에 위치한다는 가정을 전제로 한다. 만약 포인트 위치에 노이즈가 심할 경우 PDG와 PNPG의 효율성이 떨어질 가능성이 있다.

### 비판적 해석

SAM의 class-agnostic 특성을 극복하기 위해 복잡한 2단계 정제 과정을 도입한 점은 타당하나, 이는 결과적으로 pseudo-label의 품질에 크게 의존하는 구조이다. 향후 연구에서는 SAM 자체를 fine-tuning하거나 적응형 프롬프팅을 통해 정제 단계의 복잡도를 줄이는 방향으로 발전할 수 있을 것으로 보인다.

## 📌 TL;DR

**요약:**
SAPNet은 SAM의 강력한 분할 능력과 MIL 기반의 의미론적 필터링을 결합하여, 단일 포인트 주석만으로 고성능 인스턴스 분할을 수행하는 프레임워크이다. PDG와 PNPG, BMS 전략을 통해 SAM의 의미론적 모호성과 MIL의 국소적 선택 문제를 동시에 해결하였다.

**잠재적 영향:**
어노테이션 비용을 획기적으로 줄이면서도 완전 지도 학습에 근접한 성능을 낼 수 있어, 데이터 구축 비용이 매우 높은 의료 영상 분석이나 특수 목적의 영상 인식 분야에 즉시 적용 가능성이 높다.
