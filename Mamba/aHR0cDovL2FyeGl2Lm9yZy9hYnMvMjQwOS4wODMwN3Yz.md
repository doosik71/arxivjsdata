# MedSegMamba: 3D CNN-Mamba Hybrid Architecture for Brain Segmentation

Aaron Cao, Zongyu Li, Jordan Jomsky, Andrew F. Laine, and Jia Guo (2024)

## 🧩 Problem to Solve

본 논문은 MRI 스캔을 통한 뇌의 하피질(subcortical) 영역 분할(segmentation) 문제를 해결하고자 한다. 하피질 분할은 조현병, 주요 우울 장애, 치매와 같은 다양한 신경정신질환의 형태학적 결함을 감지하고 모니터링하는 데 필수적인 정량적 구조 정보를 제공한다.

그러나 기존의 분석 파이프라인은 다음과 같은 한계를 지닌다. 첫째, FreeSurfer와 같은 전통적인 도구는 정확하지만 처리 시간이 매우 오래 걸려 대규모 데이터셋에 적용하기 어렵다. 둘째, FastSurfer와 같은 딥러닝 기반의 2.5D 접근 방식은 연산 효율성은 높으나 뇌의 복잡한 3D 공간적 의존성을 완전히 캡처하지 못한다는 단점이 있다. 셋째, 3D CNN 모델은 국소적 수용역(local receptive field)으로 인해 전역적인 문맥(global context) 파악이 어렵고, Vision Transformer(ViT) 계열은 시퀀스 길이에 따라 연산 비용이 이차적으로 증가하여 고해상도 3D 의료 영상 처리에 막대한 메모리 자원을 소모한다.

따라서 본 연구의 목표는 3D 공간 정보의 효율적인 캡처와 연산 효율성을 동시에 달성하여, 정확하고 빠른 3D 하피질 뇌 분할 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CNN의 국소 특징 추출 능력과 Mamba의 선형 스케일링 전역 문맥 학습 능력을 결합한 하이브리드 아키텍처를 제안하는 것이다. 특히, 기존의 3D Mamba 모델들이 제한적인 방향으로만 데이터를 스캔했던 한계를 극복하기 위해, 3D 볼륨 데이터를 48가지의 고유한 경로로 펼쳐서 처리하는 **SS3D(3D Selective Scan)** 모듈을 설계하였다. 이를 통해 하드웨어 효율성을 유지하면서도 3D 영상의 복잡한 기하학적 구조를 정밀하게 학습할 수 있도록 하였다.

## 📎 Related Works

기존의 뇌 분할 연구는 크게 세 가지 방향으로 진행되었다.

1. **전통적/2.5D 방식**: FreeSurfer는 표준적이지만 매우 느리며, FastSurfer는 2D CNN을 활용해 속도를 높였으나 3D 공간 정보 손실이라는 한계가 있다.
2. **3D CNN 및 Transformer**: 3D U-Net 등은 기하학적 구조 파악에 유리하지만 전역 문맥 파악이 어렵다. 이를 해결하기 위해 제안된 TABSurfer와 같은 Transformer 하이브리드 모델은 전역 수용역을 갖지만, 연산 복잡도가 시퀀스 길이의 제곱에 비례하여 고해상도 영상 처리에 메모리 제약이 크다.
3. **State Space Models (Mamba)**: Mamba는 선택적 스캔(selective scan) 메커니즘을 통해 시퀀스 길이에 대해 선형적인 복잡도를 가지면서도 Transformer 수준의 성능을 낸다. Vision Mamba(Vim)나 VMamba 등이 2D 영상에 적용되었고, U-Mamba나 SegMamba가 3D 의료 영상에 적용되었으나, 이들은 스캔 방향이 제한적(1~3방향)이어서 3D 문맥을 충분히 활용하지 못한다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인

본 모델은 3D 패치 기반(patch-based) 접근 방식을 사용한다. 입력 MRI 스캔은 $256 \times 256 \times 256$ 크기로 정규화되며, 여기서 $96 \times 96 \times 96$ 크기의 패치를 16복셀 간격으로 추출하여 모델에 입력한다. 추론 단계에서는 각 패치의 예측 확률을 투표 메커니즘(voting mechanism)을 통해 결합하여 최종 라벨 맵을 재구성한다.

### 모델 아키텍처

전체 구조는 3D CNN Encoder, VSS3D Bottleneck, 3D CNN Decoder로 구성된 U-Net 형태의 하이브리드 구조이다.

1. **3D CNN Encoder & Decoder**: 4개 층의 residual block과 3회의 max pooling을 통해 다운샘플링을 수행한다. 각 residual block은 `3D Conv $\rightarrow$ Group Norm $\rightarrow$ ReLU` 순서로 구성되며, skip connection을 통해 디코더에 저차원 특징을 전달한다.
2. **SS3D (3D Selective Scan) Module**: 본 논문의 핵심 구성 요소로, 3D 볼륨을 48가지의 서로 다른 경로로 펼친다.
    - 입력 볼륨의 축을 6가지 가능한 조합($o_0 \sim o_5$)으로 전치(transpose)한다.
    - 각 조합당 8개의 시퀀스(회전 및 역방향 포함)를 추출하여 총 48개의 경로를 생성한다.
    - 각 시퀀스는 독립적인 S6 블록에 의해 병렬 처리된 후 다시 병합되어 원래의 볼륨 형태로 복원된다.
    - S6 블록의 SSM 상태 차원(state dimension)을 기존 16에서 64로 확장하여 복잡한 구조의 세부 정보를 더 잘 추출하도록 설계하였다.
3. **VSS3D (3D Visual State Space) Block**: Transformer 블록과 유사한 구조를 가지며, Self-Attention을 SS3D 모듈로 대체하였다.
    - **첫 번째 잔차 모듈**: `Layer Norm $\rightarrow$ Linear Projection $\rightarrow$ Depth-wise Conv $\rightarrow$ SiLU $\rightarrow$ SS3D $\rightarrow$ Layer Norm $\rightarrow$ Linear Projection`
    - **두 번째 잔차 모듈**: `Layer Norm $\rightarrow$ MLP`
4. **Bottleneck**: 총 9개의 VSS3D 블록이 배치되어 전역적인 문맥 정보를 학습한다.

### 학습 절차

- **손실 함수**: 첫 번째 에포크에서는 Dice Loss와 Weighted Cross Entropy를 결합하여 사용하고, 이후 에포크부터는 Dice Loss만을 사용하여 최적화한다.
- **최적화**: AdamW 옵티마이저와 Cosine Annealing Warm Restarts 스케줄러를 사용한다.
- **학습 설정**: 하피질 분할은 15 에포크, 해마 하분야(hippocampal subfield) 분할은 63 에포크 동안 학습되었다.

## 📊 Results

### 실험 설정

- **데이터셋**: 다양한 소스에서 수집된 1,784개의 T1-weighted MRI 스캔(하피질 분할)과 1,221개의 스캔(해마 하분야 분할)을 사용하였다.
- **비교 대상**: FastSurferVINN (2.5D CNN), TABSurfer (Transformer 하이브리드), SegMambaBot (Tri-oriented Mamba 기반)과 비교하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC $\uparrow$), Volume Similarity (VS $\uparrow$), Average Symmetric Surface Distance (ASSD $\downarrow$)를 사용하였다.

### 주요 결과

1. **하피질 분할 (Subcortical Segmentation)**:
    - MedSegMamba는 DSC $0.88383$, VS $0.97076$, ASSD $0.33604$로 모든 지표에서 최고 성능을 달성하였다.
    - 비-Mamba 기반 모델(FastSurfer, TABSurfer) 대비 통계적으로 유의미한 성능 향상을 보였다 ($P < 0.001$).
    - SegMambaBot과 비교했을 때 DSC와 VS는 유사했으나, 경계선 묘사 능력을 나타내는 ASSD에서 유의미한 개선을 보였으며, 파라미터 수는 약 20% 더 적었다.

2. **해마 하분야 분할 (Hippocampal Subfield Segmentation)**:
    - 매우 작고 복잡한 구조를 다루는 이 작업에서 MedSegMamba의 우위가 더 두드러졌다.
    - MedSegMamba는 유일하게 25개 모든 클래스를 성공적으로 예측한 모델이었다. (SegMambaBot은 24개, TABSurfer는 20개 클래스만 예측)
    - DSC와 VS 지표에서 다른 모델들을 압도하는 성능을 보여주었다.

## 🧠 Insights & Discussion

본 연구는 Mamba의 선택적 스캔 메커니즘을 3D 공간으로 확장하여, Transformer의 높은 메모리 비용 문제를 해결함과 동시에 CNN의 국소적 한계를 극복하였다. 특히 SS3D 모듈을 통해 48방향의 스캔 경로를 확보함으로써, 적은 파라미터 수로도 복잡한 뇌 구조의 경계선을 정밀하게 구분할 수 있음을 입증하였다.

또한, 2.5D 방식인 FastSurfer보다 3D 패치 기반 방식이 해부학적 연속성을 더 잘 보존하며, 겹치는 패치를 이용한 재구성 과정이 Ground Truth의 노이즈를 완화하여 더 매끄러운 분할 결과를 생성한다는 점을 확인하였다.

**한계점 및 비판적 해석**:
파라미터 수가 적음에도 불구하고, SS3D의 다양한 스캔 경로 처리로 인해 SegMambaBot보다 추론 속도가 다소 느리다는 점이 한계로 지적되었다. 그러나 이는 패치 추출 시 step size를 조정함으로써 실용적인 수준(스캔당 약 22초)으로 단축할 수 있다. 향후 연구에서는 SS3D 모듈을 봇틀넥뿐만 아니라 인코더의 다양한 단계에 통합하는 실험이나, 최적의 스캔 방향 개수를 결정하는 절제 연구(ablation study)가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 3D MRI 뇌 분할을 위해 CNN과 Mamba를 결합한 **MedSegMamba** 아키텍처를 제안한다. 핵심 기여는 3D 볼륨을 48가지 경로로 스캔하는 **SS3D 모듈**을 도입하여, 연산 효율성을 유지하면서도 기존 Transformer나 단순 Mamba 모델보다 정밀한 3D 공간 문맥 학습을 가능하게 한 것이다. 실험 결과, 하피질 및 해마 하분야 분할에서 기존 SOTA 모델들을 능가하는 성능을 보였으며, 특히 작은 구조물에 대한 분할 능력이 탁월하여 향후 정밀 의료 영상 분석에 중요한 역할을 할 것으로 기대된다.
