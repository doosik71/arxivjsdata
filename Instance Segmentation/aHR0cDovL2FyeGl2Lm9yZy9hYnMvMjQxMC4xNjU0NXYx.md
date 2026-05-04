# PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Model

Zhongchen Deng, Zhechen Yang, Chi Chen, Cheng Zeng, Yan Meng, Bisheng Yang (2024)

## 🧩 Problem to Solve

본 논문은 RGB-D 데이터로부터 평면 인스턴스 분할(Plane Instance Segmentation)을 수행하는 문제를 다룬다. 평면 분할은 실내 3D 재구성, 자율 주행, SLAM 등 다양한 하위 작업에서 매우 중요한 역할을 한다.

기존의 딥러닝 기반 방법론들은 주로 RGB-D 데이터셋을 사용함에도 불구하고, 실제로는 RGB 밴드(분광 데이터)의 정보만을 활용하고 Depth 밴드(기하 데이터)의 역할을 간과하는 경향이 있다. 단일 뷰 RGB 이미지에서 평면을 분할하는 것은 본질적으로 불량 설정 문제(ill-posed problem)이며, 분광 특성만으로는 3D 공간의 변화를 효과적으로 포착할 수 없어 강건성이 떨어진다는 한계가 있다.

따라서 본 연구의 목표는 RGB-D의 모든 밴드를 종합적으로 활용하는 멀티모달(Multimodal) 방식을 통해, 기하학적 정보를 충분히 반영하여 평면 인스턴스 분할의 정확도와 강건성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Dual-Complexity Backbone** 구조를 설계하여, 대규모 데이터로 사전 학습된 파운데이션 모델(EfficientSAM)의 성능을 유지하면서도 제한된 RGB-D 데이터셋에서 Depth 정보의 특징을 효과적으로 학습하는 것이다.

주요 기여 사항은 다음과 같다:
1. **Dual-Complexity Backbone 설계**: 복잡한 Transformer 브랜치(RGB용)와 단순한 CNN 브랜치(Depth용)를 병렬로 구성하여, 데이터 부족으로 인한 과적합(Overfitting)을 방지하고 모델의 일반화 성능을 높였다.
2. **불완전한 유사 라벨(Imperfect Pseudo-labels) 기반 사전 학습**: 대규모 RGB-D 데이터셋에 대해 SAM-H를 이용해 자동으로 생성된 유사 라벨을 활용하여, 비용 효율적인 자기지도 학습(Self-supervised pretraining) 전략을 제안하였다.
3. **손실 함수 최적화**: EfficientSAM이 대형 평면 분할에 취약한 점을 해결하기 위해 Focal Loss와 Dice Loss의 결합 비율을 조정하여 크기에 상관없이 효과적인 분할이 가능하도록 하였다.
4. **SOTA 달성**: ScanNet 데이터셋에서 새로운 SOTA 성능을 기록하였으며, Matterport3D, ICL-NUIM RGB-D, 2D-3D-S 데이터셋에 대한 제로샷 전이(Zero-shot transfer)에서도 우수한 성능을 입증하였다.

## 📎 Related Works

### 1. RGB-D 기반 평면 분할 연구
기존의 PlaneRCNN, PlaneAE, PlaneSeg, BT3DPR 등은 RGB-D 데이터셋을 사용하지만, 실제 모델 학습과 추론에는 RGB 밴드만을 활용하였다. PlaneTR이나 PlaneAC 같은 최신 연구들은 RGB 밴드에서 선분(Line segments)과 같은 기하학적 단서를 추출하여 보완하려 했으나, 이 역시 분광 특성에 의존하므로 촬영 조건의 변화에 민감하다는 한계가 있다.

### 2. Segment Anything Model (SAM)
SAM은 프롬프트를 통해 어떤 객체든 분할할 수 있는 강력한 제로샷 일반화 능력을 갖춘 파운데이션 모델이다. 다만, 계산 비용이 매우 높다는 단점이 있어 EfficientSAM과 같은 경량화 모델이 제안되었다. 본 논문은 EfficientSAM을 기반으로 하되, 이를 RGB-D 도메인으로 확장하기 위한 구조적 변경과 학습 전략을 적용하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인
PlaneSAM은 **Top-down** 방식을 채택하며, 크게 두 단계의 서브 네트워크로 구성된다.
1. **평면 검출 네트워크 (Plane Detection Network)**: Faster R-CNN을 사용하여 이미지 내 평면의 경계 상자(Bounding Box)를 예측한다.
2. **멀티모달 평면 마스크 생성 네트워크 (Multimodal Plane Instance Mask Generation Network)**: 예측된 경계 상자를 프롬프트로 입력받아, 수정된 EfficientSAM 구조를 통해 정밀한 평면 마스크를 생성한다.

### 2. Dual-Complexity Backbone
RGB 밴드와 Depth 밴드의 특성 차이를 극복하기 위해 서로 다른 복잡도를 가진 두 브랜치를 구성하였다.

- **Complex Branch (RGB)**: EfficientSAM의 기존 Vision Transformer(ViT) 구조를 유지한다. 대규모 RGB 데이터로 이미 학습된 강력한 특징 표현 능력을 활용한다.
- **Simple Branch (Depth)**: DPLNet의 멀티모달 프롬프트 생성 모듈과 같은 단순한 CNN 블록을 사용한다.
- **결합 방식**: EfficientSAM의 각 Encoder 블록 앞에 LWCNN(Lightweight CNN) 블록을 추가하여 RGB 특징과 Depth 특징을 다단계로 융합한다.

여기서 저자들은 RGB 브랜치의 가중치를 동결(Freeze)하지 않고 미세 조정(Fine-tuning)하였다. 그 이유는 Transformer 블록의 복잡도가 CNN 블록보다 훨씬 높아 학습 속도가 느리기 때문에, 단순 CNN 브랜치가 Depth 특징을 먼저 학습하는 동안 Transformer 브랜치는 과적합 없이 서서히 현재 작업에 적응할 수 있기 때문이다.

### 3. 손실 함수 (Loss Function)
EfficientSAM은 원래 다양한 크기의 객체를 분할하도록 설계되어 소형 객체에 편향된 경향이 있다. 이를 해결하기 위해 다음과 같이 손실 함수를 수정하였다.

기존 EfficientSAM은 $\text{Focal Loss} : \text{Dice Loss} : \text{MSE Loss} = 20 : 1 : 1$의 비율을 사용하였으나, 본 논문에서는 $\text{MSE Loss}$를 제거하고 $\text{Focal Loss}$와 $\text{Dice Loss}$를 $1 : 1$ 비율로 결합하였다.

$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{focal} + \lambda_2 \mathcal{L}_{dice} \quad (\lambda_1 = \lambda_2) $$

- **Focal Loss**: 분류하기 어려운 샘플(주로 소형 객체나 경계가 불분명한 영역)에 집중하여 소형 평면 분할 성능을 유지한다.
- **Dice Loss**: 겹침 비율을 최적화하여 클래스 불균형 문제를 해결하고, 특히 대형 평면의 분할 품질을 크게 향상시킨다.

### 4. 학습 절차
학습은 두 단계로 진행된다.
1. **사전 학습 (Pretraining)**: ScanNet 25k, SUN RGB-D, 2D-3D-S에서 수집한 약 10만 장의 RGB-D 이미지에 대해, SAM-H가 생성한 불완전한 유사 라벨을 사용하여 'Segment Anything' 작업을 수행함으로써 모델을 RGB-D 도메인에 적응시킨다.
2. **미세 조정 (Fine-tuning)**: ScanNet 데이터셋을 사용하여 실제 평면 인스턴스 분할 작업에 맞게 모델을 최적화한다. 이때 Faster R-CNN의 부정확성을 고려하여 경계 상자에 $0 \sim 10\%$의 무작위 노이즈를 추가하는 데이터 증강을 적용하였다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: ScanNet (학습 및 테스트), Matterport3D, ICL-NUIM RGB-D, 2D-3D-S (테스트 전용).
- **평가 지표**: Rand Index ($\text{RI} \uparrow$), Variation of Information ($\text{VOI} \downarrow$), Segmentation Coverage ($\text{SC} \uparrow$).

### 2. 정량적 결과
ScanNet 데이터셋 결과(Table 1)에서 PlaneSAM은 $\text{VOI}=0.550, \text{RI}=0.941, \text{SC}=0.873$을 기록하며 PlaneAE, PlaneTR, X-PDNet, BT3DPR, PlaneAC 등 기존 SOTA 모델들을 모든 지표에서 앞질렀다.

제로샷 전이 실험(Table 2)에서도 Matterport3D와 2D-3D-S 데이터셋에서 매우 경쟁력 있는 성능을 보였다. 다만, ICL-NUIM RGB-D 데이터셋에서는 $\text{VOI}$ 값이 다소 높게 나타났는데, 이는 해당 데이터셋의 Depth 이미지에 노이즈가 많아 Depth 정보를 활용하는 PlaneSAM이 영향을 받았기 때문으로 분석된다.

### 3. 효율성 분석
EfficientSAM과 비교했을 때, PlaneSAM의 학습 속도는 약 $9.49\%$ 느려졌으며, 테스트 속도는 약 $10.30\%$ 느려졌다. 즉, 계산 오버헤드는 약 $10\%$ 증가하는 수준에서 비약적인 성능 향상을 이루었다.

## 🧠 Insights & Discussion

### 1. Dual-Complexity 구조의 유효성
실험 결과, 단순히 EfficientSAM의 가중치를 동결하거나(Freeze) 단순한 단일 브랜치만 사용했을 때보다 Dual-Complexity 구조가 훨씬 우수한 성능을 보였다. 이는 파운데이션 모델이 가진 일반적인 특징 표현 능력을 유지하면서도, 데이터가 부족한 새로운 도메인(Depth 밴드)의 특징을 효율적으로 학습할 수 있는 구조적 장치가 필요함을 시사한다.

### 2. 유사 라벨 사전 학습의 가능성
정교하게 주석이 달린 데이터셋을 구축하는 것은 비용이 매우 많이 든다. 본 논문은 비록 불완전하더라도 관련 작업(Segment Anything)의 모델이 생성한 유사 라벨이 모델의 도메인 적응에 큰 도움을 준다는 것을 증명하였다. 이는 향후 3D 포인트 클라우드 분할 등 주석 작업이 어려운 분야에서도 유사한 전략을 사용할 수 있음을 보여준다.

### 3. 한계점
PlaneSAM은 Depth 이미지의 노이즈에 민감하며, 1단계인 평면 검출(Faster R-CNN)의 정확도에 의존적이다. 따라서 Depth 이미지의 노이즈 강건성을 높이고 검출 단계의 정밀도를 개선하는 것이 향후 연구 과제이다.

## 📌 TL;DR

PlaneSAM은 EfficientSAM을 기반으로 **RGB(복잡한 Transformer)와 Depth(단순 CNN) 브랜치를 병렬로 구성한 Dual-Complexity Backbone**을 통해 RGB-D 멀티모달 평면 분할을 수행하는 모델이다. 대규모 RGB-D 데이터의 **유사 라벨을 이용한 사전 학습**과 **손실 함수 비율 조정**을 통해 대형 평면 분할 성능을 극대화하였으며, 계산 비용 증가를 최소화하면서도 ScanNet 등 주요 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 파운데이션 모델을 데이터가 제한적인 새로운 센서 도메인으로 확장하는 효과적인 방법론을 제시하였다.