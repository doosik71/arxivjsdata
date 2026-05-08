# Automated Surgical Skill Assessment in Endoscopic Pituitary Surgery using Real-time Instrument Tracking on a High-fidelity Bench-top Phantom

Adrito Das et al. (2024)

## 🧩 Problem to Solve

수술 숙련도는 환자의 예후와 직결되지만, 이를 평가하는 기존의 방법은 주관적이고 노동 집약적이며 도메인 전문가의 전문 지식을 필요로 한다. 특히 내시경 접형뼈 접근법(endoscopic TransSphenoidal Approach, eTSA)은 학습 곡선이 매우 가파른 수술이다.

기존의 자동화된 수술 숙련도 평가 모델들은 주로 복강경 수술이나 로봇 수술, 또는 특정 단일 작업(isolated tasks)에 집중된 제한적인 데이터셋에서 테스트되었다. 따라서 eTSA와 같은 비복강경, 비로봇 환경에서의 수술 숙련도를 객관적으로 평가할 수 있는 자동화된 지표와 데이터셋의 필요성이 대두되었다. 본 논문의 목표는 고충실도 벤치톱 팬텀(high-fidelity bench-top phantom)을 이용한 시뮬레이션 수술 환경에서 실시간 기구 추적을 통해 수술 숙련도를 자동으로 평가하는 시스템의 가능성을 입증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **최초의 공개 데이터셋 구축**: eTSA의 비강 단계(nasal phase)를 모사한 고충실도 팬텀 수술 영상에 대해 기구 주석(annotation)과 수술 숙련도 평가 결과가 모두 포함된 최초의 공개 데이터셋을 제공한다.
2. **PRINTNet 제안**: eTSA 비강 단계의 기구를 실시간으로 분류, 세그멘테이션 및 추적할 수 있는 베이스라인 네트워크인 PRINTNet(Pituitary Real-time INstrument Tracking Network)을 구축하고, 이를 NVIDIA Clara AGX에 통합하여 실시간 성능을 구현하였다.
3. **숙련도 상관관계 분석**: 기구 추적 데이터로부터 추출한 정량적 지표와 전문가의 수술 숙련도 평가(mOSATS) 사이의 통계적 상관관계를 분석하여 자동 평가의 타당성을 제시하였다.

## 📎 Related Works

기존의 수술 숙련도 평가는 OSATS(Objective Structured Assessment of Technical Skills)와 같은 척도를 사용하였으나, 이는 평가자 간의 변동성이 크고 시간이 많이 소요되는 한계가 있다. 이를 해결하기 위해 데이터 기반의 지표를 활용한 연구들이 진행되었다.

- **복강경 및 로봇 수술**: JIGSAWS와 같은 데이터셋을 통해 단일 작업의 숙련도를 평가하거나, Mask R-CNN 및 DeepSORT를 활용해 로봇 갑상선 수술의 숙련도를 예측한 연구가 존재한다.
- **기존 방식과의 차별점**: 이전 연구들은 주로 로봇 수술이나 특정 작업에 한정되었으나, 본 연구는 비로봇, 비복강경 환경이며 기구 추적 데이터가 내장되지 않은 실제 시뮬레이션 수술 영상만을 사용한다는 점에서 차별성을 가진다. 특히 eTSA 특유의 큰 카메라 움직임, 기구의 빈번한 출입으로 인한 크기 변화, 좁은 작업 공간으로 인한 광각 렌즈 왜곡 등의 컴퓨터 비전적 난제를 다룬다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 시스템은 **[영상 입력 $\rightarrow$ PRINTNet (세그멘테이션 및 추적) $\rightarrow$ 정량적 지표 추출 $\rightarrow$ 숙련도 분류기]**의 순서로 구성된다.

### 2. PRINTNet 아키텍처 및 추적 절차

PRINTNet은 실시간 성능과 정확도를 동시에 확보하기 위해 다음과 같은 구성 요소를 사용한다.

- **세그멘테이션 (Segmentation)**:
  - **Encoder**: 사전 학습되지 않은 ResNet50을 사용하여 가볍고 강력한 특징 추출을 수행한다.
  - **Decoder**: DeepLabV3를 사용한다. 특히 Atrous Convolution(Dilation Convolution)을 적용하여 공간 해상도를 유지하면서 수용 영역(receptive field)을 넓혔다. 이는 기구가 내시경 뷰에 들어오고 나감에 따라 나타나는 다양한 크기의 기구 특징을 효과적으로 캡처하기 위함이다.
- **추적 (Tracking)**: StrongSORT를 활용한다. SORT $\rightarrow$ DeepSORT $\rightarrow$ StrongSORT로 이어지는 발전 경로를 따라, StrongSORT는 향상된 특징 추출기, 업데이트된 특징 뱅크 및 매칭 알고리즘을 사용한다. 특히 eTSA에서 발생하는 큰 카메라 움직임을 보정하기 위해 프레임 간 전역 회전 및 평행 이동을 추정하는 기능을 포함한다.
- **실시간 구현**: NVIDIA Holoscan SDK와 TensorRT 엔진을 통해 부동 소수점 정밀도 감소 및 메모리 최적화를 수행하여 NVIDIA Clara AGX에서 구동한다.

### 3. 학습 절차 및 손실 함수

- **데이터 전처리**: 4-fold 교차 검증을 수행하며, 클래스 불균형을 해소하기 위해 다수 클래스는 다운샘플링하고 소수 클래스는 업샘플링하였다.
- **손실 함수**: Cross-entropy loss를 사용하였으며, Adam 옵티마이저(learning rate $0.00006$)를 통해 최적화하였다.
- **학습 설정**: 이미지 크기를 $288 \times 512$ 픽셀로 조정하고 배치 크기는 16으로 설정하여 50 에포크(epoch) 동안 학습하였다.

### 4. 수술 숙련도 평가 모델

추적 데이터로부터 시간, 움직임, 사용량과 관련된 34개의 지표를 추출한다.

- **분석 방법**: 각 지표와 mOSATS(Modified OSATS) 점수 간의 피어슨 상관 계수(Pearson Correlation Coefficient, PCC)를 계산한다.
- **분류 모델**: 추출된 지표를 입력으로 하여 Linear, SVM, Random Forest, MultiLayer Perceptron(MLP) 모델을 통해 숙련도(초보자 vs 전문가)를 분류한다.

## 📊 Results

### 1. 기구 세그멘테이션 및 추적 성능

- **세그멘테이션**: DeepLabV3가 전반적으로 가장 높은 mIoU를 기록하였다. 특히 Blunt Dissector와 Kerrisons 클래스에서 높은 성능을 보였으나, Cup Forceps와 Pituitary Ronguers는 데이터 불균형으로 인해 상대적으로 낮은 성능을 보였다.
- **추적 성능**: StrongSORT가 카메라 움직임을 보정함으로써 가장 높은 MOTP(Multiple Object Tracking Precision)인 $71.9\%$를 달성하였다.
- **실시간성**: NVIDIA Clara AGX에서 FP16 정밀도로 실행했을 때 $22\text{ FPS}$의 속도와 $100\text{ms}$의 지연 시간을 기록하여 실시간 피드백이 가능함을 입증하였다.

### 2. 수술 숙련도 예측 결과

- **이진 분류 (Binary-class)**: 초보자와 전문가를 구분하는 작업에서 MLP 모델이 $87\%$의 높은 정확도를 달성하였다.
- **다중 클래스 분류 (Multi-class)**: 평균 mOSATS 점수를 예측하는 작업에서는 정확도가 상대적으로 낮게 나타나, 이 문제의 복잡성과 더 많은 데이터의 필요성을 시사하였다.
- **주요 지표**: '전체 수술 시간 대비 기구 가시 시간의 비율(ratio of total procedure time to instrument visible time)'이 수술 숙련도와 정적 상관관계를 보였다. 즉, 기구의 유휴 시간이 적고 효율적으로 사용하는 것이 높은 숙련도와 연결된다.

## 🧠 Insights & Discussion

### 1. 강점 및 발견

본 연구는 eTSA라는 특수한 환경에서 실시간 기구 추적을 통한 숙련도 평가의 가능성을 확인하였다. 특히 움직임 기반 지표보다 시간 기반 지표가 숙련도 예측에 더 강력한 변수라는 점을 발견하였다. 이는 로봇 수술 연구에서는 움직임(economy of motion)이 중요했던 것과 대조적인 결과이다.

### 2. 한계 및 분석

- **카메라 움직임의 영향**: eTSA의 비강 단계에서는 콧구멍을 통과하기 위해 필연적으로 큰 내시경 움직임이 발생한다. 이러한 전역적 움직임이 기구의 미세한 움직임보다 훨씬 크기 때문에, 단순한 움직임 기반 지표만으로는 숙련도를 구분하기 어렵다.
- **데이터 불균형**: 기구의 핸들 부분은 서로 유사하고 왜곡으로 인해 이미지의 큰 부분을 차지하는 반면, 기구의 팁(tip) 부분은 매우 작다. 이로 인해 세그멘테이션 성능이 팁 부분에서 저하되는 경향이 있으며, 이는 분류 정확도에 영향을 미친다.
- **데이터셋 규모**: 15개의 영상만으로 학습되었기에 모델의 일반화 성능에 한계가 있으며, 다중 클래스 분류 성능이 낮게 나타난 주된 원인으로 분석된다.

## 📌 TL;DR

본 논문은 내시경 접형뼈 수술(eTSA)의 시뮬레이션 환경을 위한 최초의 공개 데이터셋을 구축하고, 이를 기반으로 실시간 기구 추적 네트워크인 **PRINTNet**을 제안하였다. PRINTNet은 DeepLabV3와 StrongSORT를 결합하여 실시간(22 FPS)으로 기구를 추적하며, 여기서 추출된 시간 효율성 지표를 통해 수술자의 숙련도를 $87\%$의 정확도로 구분해낼 수 있다. 이 연구는 향후 수술 교육 과정에서 초보 의사들에게 객관적인 실시간 피드백을 제공하는 시스템의 기반이 될 것으로 기대된다.
