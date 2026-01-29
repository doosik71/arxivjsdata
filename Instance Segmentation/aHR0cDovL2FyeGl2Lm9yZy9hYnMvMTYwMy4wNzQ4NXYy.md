# Simple Does It: Weakly Supervised Instance and Semantic Segmentation

Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele

## 🧩 Problem to Solve

픽셀 단위의 정밀한 어노테이션(semantic labelling 및 instance segmentation)은 매우 비용이 많이 들고 시간 소모적인 작업($\sim15$배 더 오래 걸림)입니다. 이 연구의 주요 목표는 비교적 저렴하고 쉽게 얻을 수 있는 바운딩 박스 어노테이션만을 사용하여, 기존의 완전 지도(fully supervised) 모델에 필적하는 고품질 semantic segmentation 및 instance segmentation을 달성하는 방법을 탐구하는 것입니다. 즉, 대규모 픽셀 단위 어노테이션의 필요성을 줄이는 것입니다.

## ✨ Key Contributions

* **재귀적 학습(Recursive Training) 탐색**: 컨브넷(convnet)의 재귀적 학습을 통한 약지도(weakly supervised) semantic labelling 방법을 탐구하고, 고품질 결과를 얻는 방법과 접근 방식의 한계를 논의합니다.
* **훈련 절차 변경 없이 SOTA 달성**: segmentation convnet 훈련 절차를 수정하지 않고, 주어진 바운딩 박스에서 학습 레이블을 생성하기 위해 GrabCut과 같은 알고리즘을 적절히 사용하여 최신(state-of-the-art) 품질을 달성할 수 있음을 보여줍니다.
* **바운딩 박스만 사용한 최고 성능 기록**: Pascal VOC12 및 VOC12+COCO 학습 데이터를 사용하여 바운딩 박스 어노테이션만으로 훈련했을 때, 완전 지도 방식과 비교 가능한 품질을 달성하며 알려진 최고 성능을 보고합니다.
* **약지도 Instance Segmentation 최초 보고**: 약지도 instance segmentation 작업에서도 유사한 결과를 달성할 수 있음을 최초로 보여줍니다.

## 📎 Related Works

* **Semantic Labelling**: DeepLab [5]과 같은 컨브넷 기반 방법론이 지배적이며, DenseCRF [20] 같은 후처리 기법이 성능 향상에 기여합니다.
* **Weakly Supervised Semantic Labelling**: 이미지 레이블, 포인트, 스크라이블, 그리고 바운딩 박스 [8, 27] 등 다양한 형태의 약지도 학습이 연구되었습니다. BoxSup [8]은 컨브넷 예측을 다음 학습 라운드에 사용하는 재귀적 훈련을, WSSL [27]은 기댓값-최대화(expectation-maximisation) 알고리즘을 제안합니다. 본 논문은 이들과 달리 훈련 절차를 변경하지 않고 입력 레이블 생성에 집중합니다.
* **Instance Segmentation**: MCG [34]와 같은 객체 제안(object proposal) 기법이나 GrabCut [36] 변형을 사용하여 픽셀 그룹을 객체 인스턴스로 분류합니다. DeepMask [33]와 같은 컨브넷 기반 end-to-end 학습도 제안되었습니다. 본 연구는 DeepMask와 DeepLabv2 네트워크를 약지도 인스턴스 분할에 활용합니다.

## 🛠️ Methodology

본 논문은 객체 바운딩 박스 어노테이션에서 고품질 semantic labelling을 생성하는 데 초점을 맞추며, 정보 활용을 극대화하기 위해 다음과 같은 단서를 통합합니다:

* **C1 (배경)**: 박스로 덮이지 않은 모든 픽셀은 배경으로 레이블링됩니다.
* **C2 (객체 범위)**: 박스는 각 인스턴스의 범위를 정의하며, 객체 모양에 대한 사전 지식을 활용하여 예상 객체 영역 정보를 제공합니다.
* **C3 (객체성)**: 공간 연속성, 배경과의 대비 경계 등 추가 객체 사전 지식을 segment proposal 기법 [35]을 통해 활용합니다.

**접근 방식**:

1. **박스 베이스라인 (Box Baselines)**:
    * **Naive**: 바운딩 박스 내부 픽셀을 해당 클래스로 채웁니다 (겹칠 경우 작은 박스가 앞에 있다고 가정). 외부 픽셀은 배경입니다. 이 레이블로 표준 DeepLabv1 네트워크를 훈련합니다.
    * **재귀적 훈련 (Recursive Training)**: Naive 방식의 결과 모델이 객체 모양을 더 잘 포착하는 것을 관찰하여, 컨브넷 예측을 다음 라운드의 ground truth로 사용하여 여러 라운드에 걸쳐 재귀적으로 훈련합니다.
    * **Box**: Naive 재귀적 훈련에 다음 세 가지 후처리 단계를 추가하여 레이블을 개선합니다:
        * 박스 강제(Box enforcing): 박스 외부 픽셀은 배경으로 리셋 (C1).
        * 이상치 리셋(Outliers reset): 세그먼트 면적이 너무 작으면 (예: $IoU < 50\%$) 박스 영역을 초기 레이블로 리셋 (C2).
        * CRF: DenseCRF [20]를 사용하여 이미지 경계를 더 잘 따르도록 네트워크 출력을 필터링 (C3).
    * **Box$_i$**: 초기 레이블로 박스 내부의 20%만 채우고 나머지 내부 영역은 무시(ignore) 영역으로 남겨둡니다 (낮은 리콜(recall) 대신 높은 정밀도(precision)를 추구). 이후 Box와 동일한 재귀적 훈련 절차를 사용합니다.

2. **박스 기반 세그먼트 (Box-driven segments, 단일 라운드)**:
    * **GrabCut+**: HED 경계 [43]를 쌍별 항(pairwise term)으로 사용하는 GrabCut [36]의 수정 버전을 제안합니다.
    * **GrabCut+$_i$**: 여러 (약 150개) GrabCut+ 출력을 교란(jittering)하여 생성합니다. 70% 이상이 전경으로 표시하면 전경, 20% 미만이면 배경, 그 외는 무시 영역으로 설정합니다.
    * **MCG**: MCG [34] (객체 제안)를 사용하여 박스 어노테이션과 가장 많이 겹치는 제안을 해당 세그먼트로 선택합니다.
    * **M∩G+ (MCG∩GrabCut+)**: 최종 접근 방식. 어노테이션된 박스 내에서 MCG와 GrabCut+가 모두 동의하는 픽셀은 전경으로 표시하고, 나머지는 무시 영역으로 표시합니다. 이는 GrabCut+$_i$의 개선된 버전으로, 정밀도와 리콜 간의 균형을 제공합니다. 이 접근 방식은 세그멘테이션 컨브넷 훈련 절차를 변경하지 않고 입력 레이블만 생성하는 데 중점을 둡니다.

## 📊 Results

* **Semantic Labelling**:
  * **성능**: M∩G+는 Pascal VOC12 검증 세트에서 $65.7\%$ mIoU를 달성하여, 완전 지도 모델($69.1\%$ mIoU) 품질의 $\sim95\%$에 도달합니다. Box$_i$도 $62.7\%$ mIoU로 경쟁력 있는 성능을 보였습니다.
  * **학습 데이터 확장**: COCO 데이터셋을 추가하여 VOC12+COCO로 훈련했을 때, M∩G+는 $68.9\%$ mIoU를 달성하며, 이는 VOC12만으로 훈련한 완전 지도 모델과 거의 동일한 성능입니다.
  * **다양한 컨브넷 적용**: DeepLabv2-ResNet101 네트워크에서도 M∩G+는 완전 지도 모델 성능의 $93\%\sim95\%$를 달성하며, COCO 데이터와 함께 훈련 시 VOC12만으로 훈련한 완전 지도 모델과 유사한 품질을 보입니다.
* **Instance Segmentation (최초 보고)**:
  * **성능**: DeepMask 및 재목적화된 DeepLabv2 네트워크(DeepLab$_{\text{BOX}}$)를 사용하여 약지도 instance segmentation을 수행했습니다. DeepLab$_{\text{BOX}}$는 VOC12 검증 세트에서 $44.8\%$ mAP$_{r}0.5$를 달성하여, 완전 지도 모델($47.5\%$ mAP$_{r}0.5$) 품질의 $\sim95\%$에 도달합니다.
  * **결론**: 두 가지 작업 모두에서 약지도 접근 방식이 완전 지도 모델 품질의 $\sim95\%$에 도달하는 우수한 결과를 보였습니다.

## 🧠 Insights & Discussion

* **어노테이션 비용 절감**: 이 연구는 픽셀 단위 마스크 대신 바운딩 박스 어노테이션만으로도 완전 지도 모델에 준하는 고품질의 semantic 및 instance segmentation을 달성할 수 있음을 입증하여, 어노테이션 노력과 비용을 크게 줄일 수 있는 가능성을 제시합니다.
* **레이블 생성의 중요성**: 컨브넷 훈련 절차를 수정하지 않고도, 주어진 바운딩 박스에서 학습 레이블을 신중하게 설계하고 생성하는 것이 핵심적인 성공 요인입니다 (특히 M∩G+ 방식). 잡음이 있는 학습 세그먼트에서 정확도와 리콜 간의 좋은 균형을 맞추는 것이 중요합니다.
* **재귀적 훈련의 효과**: Box$_i$와 같은 재귀적 훈련 방식은 간단하지만 놀랍도록 효과적이며, 적절한 후처리 단계를 통해 출력 잡음을 제거하는 것이 중요합니다.
* **약지도 Instance Segmentation의 가능성**: 이 연구는 약지도 instance segmentation에 대한 첫 번째 결과를 보고하며, 이 분야에서의 추가 연구 가능성을 열었습니다.
* **한계 및 미래 방향**: 현재 접근 방식은 각 어노테이션된 박스를 개별적으로 처리합니다. 향후 연구에서는 co-segmentation(어노테이션 세트 전체를 통합적으로 처리) 아이디어와 더 약한 형태의 지도 학습(supervision)을 탐색할 계획입니다.

## 📌 TL;DR

픽셀 단위 어노테이션의 높은 비용 문제를 해결하기 위해, 본 논문은 바운딩 박스 어노테이션만을 사용하여 semantic segmentation 및 instance segmentation을 위한 약지도 학습 접근 방식을 제안합니다. 컨브넷 훈련 절차를 수정하지 않고, 재귀적 훈련(Box$_i$) 및 GrabCut+와 MCG 기반의 M∩G+와 같은 정교한 레이블 생성 기법(무시 영역 활용)을 통해 고품질의 학습 레이블을 만듭니다. 결과적으로, 완전 지도 모델 품질의 $\sim95\%$에 해당하는 성능을 달성하며, 약지도 instance segmentation 결과를 최초로 보고합니다. 이는 어노테이션 비용을 크게 줄이면서도 효과적인 학습이 가능함을 보여줍니다.
