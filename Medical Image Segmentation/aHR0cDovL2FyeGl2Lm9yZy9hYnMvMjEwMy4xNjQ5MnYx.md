# Assessing the Role of Random Forests in Medical Image Segmentation

Dennis HARTMANN, Dominik MÜLLER, Iñaki SOTO-REY, and Frank KRAMER (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝, 특히 깊은 합성곱 신경망(Deep Convolutional Neural Networks, DCNNs)은 인간 방사선 전문의에 필적하는 매우 높은 성능을 보여주며 표준으로 자리 잡았다. 그러나 이러한 DCNN 모델들은 방대한 양의 데이터를 처리하기 위해 고가의 고성능 GPU 하드웨어가 필수적이다. 모든 의료 현장에서 이러한 고사양 하드웨어를 갖추는 것은 현실적으로 어려움이 따를 수 있다.

본 논문은 GPU 없이도 높은 성능을 낼 수 있는 대안으로 Random Forest(RF)의 가능성을 평가하고자 한다. 연구의 핵심 목표는 두 가지 서로 다른 RF 파이프라인과 최신 DCNN 모델의 성능 및 사용성을 비교 분석하여, RF가 실제 임상 환경에서 GPU의 대안으로 사용될 수 있는지 확인하는 것이다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 이미지 분할 작업을 위해 RF를 적용하는 두 가지 서로 다른 아키텍처를 설계하고, 이를 U-Net 기반의 DCNN과 비교하는 것이다. 

1. **Feature Extraction Architecture (RF-FE)**: 픽셀 단위로 특징을 추출하여 학습하는 접근 방식을 통해 RF의 분할 성능을 극대화하고자 하였다.
2. **Whole Image Architecture (RF-WI)**: 이미지 전체를 하나의 특징 배열로 처리하는 단순한 접근 방식을 통해 RF의 기본 성능을 측정하고자 하였다.

이를 통해 GPU가 없는 환경에서도 적절한 특징 추출 기법(Feature Extraction)이 동반된다면 RF가 DCNN에 근접한 성능을 낼 수 있다는 가능성을 제시한다.

## 📎 Related Works

논문에서는 DCNN이 의료 영상 분석에서 최첨단(state-of-the-art) 성능을 보이고 있음을 언급하며, 그 이유를 효율적인 관심 영역(ROI) 인식과 일반적인 패턴 인식 능력에서 찾고 있다. 

기존의 RF 관련 연구로는 초음파 영상의 뼈 분할(Baka et al.), BRATS 데이터셋의 뇌 병변 분할(Chen et al.), 전자현미경 영상의 사구체 기저막 분할(Cao et al.) 등이 언급된다. 이러한 선행 연구들은 RF가 특정 의료 영상 과제에서 DCNN과 대등한 성과를 낼 수 있음을 시사한다. 하지만 본 논문은 RF와 DCNN을 성능과 사용성 측면에서 직접적으로 비교한 연구가 부족하다는 점을 차별점으로 내세운다.

## 🛠️ Methodology

### 데이터셋 및 전처리
연구에는 두 가지 데이터셋이 사용되었다.
- **PhC-C2DH-U373**: Glioblastoma-astrocytoma U373 세포 영상으로, 세포 분할이 목표이다.
- **Retinal Imaging Dataset**: 망막의 혈관 분할을 목표로 하는 안저 사진 데이터셋이다.

전처리를 위해 모든 이미지는 그레이스케일로 변환되었으며, `Albumentations` 라이브러리를 통해 밝기 조절, 회전, 감마 변경, 탄성 변형(elastic transform), 광학 왜곡, 플립, 대비 변경 등을 적용하여 데이터 양을 10배로 증강하였다. 이후 모든 이미지는 $512 \times 512$ 픽셀로 리사이징되었으며, RF 모델의 입력을 위해 Sobel 필터를 적용하여 엣지 정보를 추출하였다.

### Random Forest 모델
RF 모델은 100개의 트리와 최대 깊이 40으로 설정되었으며, 분할 품질 측정을 위해 cross-entropy를 사용하였다.

1. **Feature Extraction Architecture (RF-FE)**: 픽셀 단위 학습 방식을 채택하였으며, 픽셀당 4개의 특징을 사용한다.
   - 원본 이미지의 픽셀 강도(intensity)
   - 원본 이미지에서 해당 픽셀 주변 $13 \times 13$ 영역의 평균값
   - Sobel 필터링된 이미지에서 해당 픽셀 주변 $13 \times 13$ 영역의 평균값
   - 원본 이미지의 $13 \times 13$ 주변 픽셀 값들의 집합

2. **Whole Image Architecture (RF-WI)**: Sobel 필터링된 이미지 전체를 하나의 특징 배열(feature array)로 정의하여 모델의 입력으로 사용한다.

### Deep Convolutional Neural Network 모델
`MIScnn` 프레임워크를 사용하여 U-Net 아키텍처를 구현하였다.
- **학습 설정**: Batch Normalization 적용, 100 epoch 학습, Batch size 2.
- **손실 함수**: Tversky index와 Categorical Cross-Entropy의 합을 사용하였다.
- **최적화**: Adam optimizer를 사용하였으며, 학습률은 $0.001$에서 시작하여 5 epoch 동안 손실이 감소하지 않을 경우 $0.1$씩 감소하여 최소 $0.00001$까지 떨어지도록 동적으로 설정하였다.
- **조기 종료**: 10 epoch 동안 손실 개선이 없을 경우 Early Stopping을 적용하였다.

## 📊 Results

### 하드웨어 자원 소모 (Table 1)
| 알고리즘 | 메모리(RAM) | GPU 사용 여부 | 학습 시간 |
| :--- | :--- | :--- | :--- |
| RF-FE | 207 GB | X | 6.79h |
| RF-WI | $>220$ GB | X | 0.71h |
| DCNN | 3 GB | 7 GB | 0.43h |

RF 방식은 GPU를 사용하지 않지만, 매우 방대한 양의 시스템 메모리(RAM)를 요구하며 학습 시간이 DCNN보다 길다.

### 분할 성능 평가 (Table 2)
평가 지표로는 Accuracy, Dice similarity coefficient, IoU(Intersection-over-Union), Sensitivity가 사용되었다.

- **PhC-C2DH-U373 (세포 분할)**: 
  - DCNN이 Dice 0.90으로 가장 우수한 성능을 보였으며, RF-FE가 Dice 0.85로 그 뒤를 이었다.
  - RF-WI는 Dice 0.23으로 매우 낮은 성능을 보였다.
- **Retinal Imaging (혈관 분할)**:
  - DCNN이 Dice 0.77로 가장 높았고, RF-FE는 Dice 0.68을 기록하였다.
  - RF-WI는 Dice 0.00으로 사실상 분할에 실패하였다.

## 🧠 Insights & Discussion

### 분석 및 해석
전반적인 정확도와 민감도 측면에서 DCNN이 RF보다 우수한 성능을 보였다. 이는 신경망 아키텍처가 이미지 내의 일반적인 패턴을 인식하고 관심 영역을 더 효율적으로 포착하는 능력이 뛰어나기 때문으로 풀이된다. 특히 RF-WI 방식은 복잡한 구조(예: 망막 혈관)를 분할하는 데 완전히 실패함으로써, 단순한 이미지 입력만으로는 RF의 한계가 명확함을 보여주었다.

반면, RF-FE는 DCNN과 비교하여 어느 정도 경쟁력 있는 성능을 보여주었다. 이는 정교하게 설계된 특징 추출 과정이 RF의 성능을 크게 향상시킬 수 있음을 의미한다.

### 강점과 한계
- **강점**: RF-FE는 GPU 클러스터가 없는 소규모 IT 인프라(일반 병원 환경 등)에서도 CPU와 충분한 RAM만 있다면 실행 가능하다는 유연성을 가진다.
- **한계**: 최적의 성능을 내기 위해서는 매우 높은 하드웨어 메모리(RAM)가 필요하다는 점이 치명적인 제약 사항이다. 또한, DCNN에 비해 수동으로 특징(Feature)을 설계해야 하는 번거로움이 있다.

## 📌 TL;DR

본 논문은 GPU 없이 의료 영상 분할을 수행하기 위해 Random Forest(RF)의 효용성을 분석하였다. 실험 결과, 단순한 RF 모델(RF-WI)은 성능이 매우 낮았으나, 픽셀 단위의 특징 추출을 적용한 모델(RF-FE)은 최신 DCNN(U-Net)에 근접하는 준수한 성능을 보였다. 결론적으로 RF-FE는 GPU를 사용할 수 없는 환경에서 유용한 대안이 될 수 있지만, 막대한 양의 RAM이 필요하다는 하드웨어적 제약이 존재한다. 이 연구는 향후 GPU 의존도를 낮춘 의료 영상 분석 파이프라인 구축에 참고가 될 수 있다.