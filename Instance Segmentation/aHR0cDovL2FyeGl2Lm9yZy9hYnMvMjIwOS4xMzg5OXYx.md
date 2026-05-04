# Strong Instance Segmentation Pipeline for MMSports Challenge

Bo Yan, Fengliang Qi, Zhuang Li, Yadong Li, Hongbin Wang (2022)

## 🧩 Problem to Solve

본 논문은 ACM MMSports2022 DeepSportRadar Instance Segmentation Challenge에서 제시한 과제, 즉 농구 코트 위에서 선수, 코치, 심판 등 개별 사람을 정밀하게 분할(segmentation)하는 문제를 해결하고자 한다.

이 문제는 다음과 같은 두 가지 주요한 어려움을 가지고 있다. 첫째, 경기 특성상 선수들 간의 겹침(occlusion) 현상이 매우 심하여 개별 인스턴스를 정확히 구분하기 어렵다. 둘째, 학습에 사용할 수 있는 데이터의 양이 매우 제한적이어서 모델이 과적합(overfitting)될 가능성이 높고 일반화 성능을 확보하기 어렵다는 점이다. 따라서 본 연구의 목표는 데이터 부족과 심한 폐색 문제를 극복하여 높은 성능을 내는 강력한 Instance Segmentation 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최신 고성능 백본 네트워크와 검출기 구조를 결합하고, 데이터 효율성을 극대화하는 증강 기법 및 학습 전략을 적용하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **데이터 분포 확장**: Photometric distortion 및 Copy-Paste 전략을 통해 데이터 부족 문제를 해결하고, 다양한 상황에서의 인스턴스 분포를 생성하여 모델의 강건성을 높였다.
2. **고성능 모델 아키텍처 구성**: Swin-Base 기반의 CBNetV2 백본과 CBFPN, 그리고 Hybrid Task Cascade(HTC) 검출기를 결합하여 강력한 특징 추출 및 분할 능력을 확보하였다.
3. **Mask 품질 정렬**: MaskIoU head를 HTC의 Mask head에 추가함으로써, 단순한 분류 신뢰도(classification confidence)가 아닌 실제 마스크의 품질에 기반하여 스코어링을 수행하도록 개선하였다.
4. **최적화된 학습 및 추론 전략**: SWA(Stochastic Weight Averaging)를 통한 파인튜닝과 다중 스케일 TTA(Test Time Augmentation)를 적용하여 최종 성능을 끌어올렸다.

## 📎 Related Works

논문에서는 Instance Segmentation의 기반이 되는 Mask R-CNN, PANet, TensorMask, CenterMask 및 SOLO 시리즈와 같은 딥러닝 기반 방법론들을 언급한다. 또한, 최근 컴퓨터 비전 분야에서 비지역적(non-local) 특성과 관계적 특성을 잘 포착하는 Transformer 구조, 특히 Swin Transformer가 COCO 데이터셋의 검출 및 분할 작업에서 성공적인 결과를 거두었음을 설명한다.

본 연구는 이러한 기존의 고성능 프레임워크들을 단순히 사용하는 것에 그치지 않고, 스포츠 경기라는 특수한 도메인(심한 폐색, 데이터 부족)에 맞게 데이터 증강과 마스크 스코어링 메커니즘을 최적화하여 차별점을 두었다.

## 🛠️ Methodology

제안된 파이프라인은 데이터 증강, 모델 구조, 학습 전략의 세 가지 단계로 구성된다.

### 1. 데이터 증강 (Data Augmentation)

데이터 부족과 폐색 문제를 해결하기 위해 두 가지 전략을 사용한다.

- **Photometric Distortion**: 이미지의 밝기(brightness), 대비(contrast), 채도(saturation), 색조(hue)를 무작위로 변경하여 조명 변화에 강건한 모델을 만든다.
- **Copy-Paste**: 한 이미지 내의 객체를 복사하여 다른 이미지에 붙여넣는 방식으로, 인스턴스의 수를 인위적으로 늘리고 다양한 배치 상황을 생성하여 폐색 문제 대응 능력을 키운다.

### 2. 세그멘테이션 모델 (Segmentation Model)

전체 아키텍처는 다음과 같은 구성 요소로 이루어져 있다.

- **Backbone & FPN**: Swin-Base를 기반으로 하는 $CBNetV2$ 백본과 $CBFPN$을 사용하여 다중 스케일의 풍부한 특징 맵을 추출한다.
- **Detector**: $Hybrid\ Task\ Cascade\ (HTC)$ 구조를 채택하여 검출과 분할 단계의 상호작용을 강화하였다.
- **MaskIoU Head**: 기존의 Mask R-CNN 계열은 Bounding Box의 분류 점수를 마스크의 품질로 간주하는 경향이 있다. 이를 개선하기 위해 Mask Scoring R-CNN의 개념을 도입하여, 마스크 자체의 품질($IoU$)을 예측하는 $MaskIoU$ head를 $HTCMaskHead$에 추가하였다. 이를 통해 마스크 품질과 점수 사이의 정렬(alignment)을 통해 정밀도를 높였다.

### 3. 학습 및 추론 전략 (Training & Inference Strategy)

- **학습 절차**: AdamW 옵티마이저를 사용하며 초기 학습률(learning rate)은 $0.0001$로 설정하였다. 입력 이미지는 짧은 변 기준 $820$에서 $3080$까지 무작위 스케일링 후 $(1920, 1440)$ 크기로 크롭 및 패딩 처리된다.
- **SWA (Stochastic Weight Averaging)**: 모델이 수렴한 후, SWA 전략을 사용하여 가중치를 평균화함으로써 일반화 성능을 개선하고 더 강건한 모델을 생성한다.
- **TTA (Test Time Augmentation)**: 추론 단계에서 수평 뒤집기(horizontal flip)와 5가지 스케일 팩터 $(1.0, 1.5, 2.0, 2.5, 3.0)$를 적용한 Multi-scale test를 수행하여 최종 결과를 도출한다.

## 📊 Results

### 실험 설정

- **데이터셋**: DeepSportRadar Instance Segmentation Challenge 제공 데이터.
- **평가 지표**: $AP@0.50:0.95$ (Average Precision).
- **기준선(Baseline)**: HTC-CBSwinBase 모델에 Soft-NMS를 적용한 상태.

### 주요 결과

최종적으로 챌린지 세트에서 $0.768\ AP$를 달성하였다. 상세한 성능 향상 과정은 다음과 같은 Ablation Study를 통해 확인되었다.

| 방법론 (Pipeline) | 데이터 | 기타 설정 | mAP-Test | mAP-Challenge |
| :--- | :--- | :--- | :---: | :---: |
| HTC-CBSwinBase | Train+Val | - | 0.734 | - |
| + MaskIoU | Train+Val | - | 0.744 | - |
| + MaskIoU | Train+Val | Mask Loss Weight=2.0 | 0.755 | - |
| + MaskIoU + SWA | Train+Val | Mask Loss Weight=2.0 | 0.763 | - |
| + MaskIoU + SWA + TTA | Train+Val | Mask Loss Weight=2.0 | 0.788 | 0.766 |
| + MaskIoU + SWA + TTA | Train+Val+Test | Mask Loss Weight=2.0 | - | **0.768** |

실험 결과, $MaskIoU$ 도입($+0.010$)과 Mask Loss 가중치 상향($+0.011$), SWA 적용($+0.008$) 모두 성능 향상에 기여하였으며, 특히 TTA 적용이 $+0.025\ mAP$라는 가장 큰 성능 향상을 가져왔음을 알 수 있다. 또한, 테스트 세트까지 포함하여 학습했을 때 최종 성능이 극대화되었다.

## 🧠 Insights & Discussion

본 논문은 특정 도메인의 챌린지를 해결하기 위해 개별적인 알고리즘의 혁신보다는, 검증된 고성능 컴포넌트들을 정밀하게 조합하고 최적화하는 파이프라인 구축의 중요성을 보여준다.

**강점**:

- 데이터 부족 문제를 해결하기 위해 단순 증강이 아닌 Copy-Paste와 같은 인스턴스 중심의 증강 전략을 적절히 사용하였다.
- MaskIoU head를 통해 마스크의 정밀도를 직접적으로 제어함으로써 Instance Segmentation의 고질적인 문제인 '점수-품질 불일치' 문제를 해결하였다.
- TTA와 SWA 같은 학습/추론 기법이 실제 챌린지 환경에서 상당한 성능 향상을 이끌어낼 수 있음을 정량적으로 입증하였다.

**한계 및 논의사항**:

- 논문에서 최종 성능 향상을 위해 Train, Val, Test 데이터셋 전체를 사용하여 학습하였다고 명시되어 있다. 이는 실전 챌린지에서는 가능하나, 실제 환경(unseen data)에서의 일반화 성능에 대해서는 추가적인 검증이 필요하다.
- 제안된 파이프라인이 매우 무거운 모델(Swin-Base, HTC, Multi-scale TTA)을 기반으로 하고 있어, 실시간성이 중요한 스포츠 중계 시스템에 적용하기에는 연산 비용이 매우 높을 것으로 판단된다.

## 📌 TL;DR

본 연구는 데이터 부족과 심한 폐색이 특징인 농구 경기 인스턴스 분할 문제를 해결하기 위해 **CBNetV2-HTC** 기반의 모델에 **Copy-Paste 증강**, **MaskIoU head**, **SWA 학습**, 그리고 **Multi-scale TTA**를 결합한 강력한 파이프라인을 제안하였다. 이를 통해 DeepSportRadar 챌린지에서 $0.768\ AP$라는 경쟁력 있는 성적을 거두었으며, 특히 데이터 증강과 TTA가 성능 향상에 결정적인 역할을 함을 확인하였다. 이 연구는 데이터가 제한적인 특수 도메인에서 최신 딥러닝 아키텍처를 어떻게 최적화하여 적용할 수 있는지에 대한 실무적인 가이드라인을 제공한다.
