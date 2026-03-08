#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
右臂（can1）回零脚本
基于官方 piper_ctrl_go_zero.py 修改
"""
import time
import sys
sys.path.insert(0, '/home/robo/git/X-VLA/piper_sdk')
from piper_sdk import C_PiperInterface

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 Piper 右臂回零脚本")
    print("=" * 70)
    print("⚠️  警告: 此操作将使右臂（can0）回到零位")
    print("   请确保:")
    print("   1. 机械臂周围无障碍物")
    print("   2. 有足够的运动空间")
    print("   3. 随时准备按急停按钮")
    print("=" * 70)
    
    response = input("\n是否继续? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ 操作取消")
        sys.exit(0)
    
    print("\n🔌 连接右臂 (can1)...")
    
    piper = C_PiperInterface(
        can_name="can1",
        judge_flag=False,
        can_auto_init=True,
        dh_is_offset=1,
        start_sdk_joint_limit=True,
        start_sdk_gripper_limit=True
    )
    
    piper.ConnectPort()
    time.sleep(0.5)
    
    print("⚡ 使能机械臂...")
    piper.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.2)
    piper.EnableArm(7, 0x02)
    time.sleep(0.2)
    
    print("🔧 使能夹爪...")
    try:
        piper.GripperCtrl(0, 1000, 0x02, 0)
        time.sleep(0.1)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.2)
        print("✅ 夹爪使能成功")
    except Exception as e:
        print(f"⚠️  夹爪使能警告: {e}")
    
    print("\n🎯 发送回零指令...")
    print("   目标位置: 所有关节角度 = 0°")
    print("   速度: 30%")
    
    factor = 57295.7795  # 1000*180/3.1415926 (弧度转0.001度)
    position = [0, 0, 0, 0, 0, 0, 0]  # 零位
    
    joint_0 = round(position[0] * factor)
    joint_1 = round(position[1] * factor)
    joint_2 = round(position[2] * factor)
    joint_3 = round(position[3] * factor)
    joint_4 = round(position[4] * factor)
    joint_5 = round(position[5] * factor)
    joint_6 = round(position[6] * 1000 * 1000)  # 夹爪位置(微米)
    
    # 设置运动模式
    piper.ModeCtrl(0x01, 0x01, 30, 0x00)
    time.sleep(0.1)
    
    # 发送关节控制指令
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    time.sleep(0.1)
    
    # 夹爪回零
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    
    print("✅ 回零指令已发送")
    print("\n⏳ 等待机械臂运动到零位...")
    print("   (运动过程中可按 Ctrl+C 停止)")
    
    try:
        # 等待足够时间让机械臂完成运动
        time.sleep(10)
        print("\n✅ 右臂已回零")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  用户中断")
    
    finally:
        print("\n🔌 断开连接...")
        piper.DisconnectPort()
        print("✅ 完成")
