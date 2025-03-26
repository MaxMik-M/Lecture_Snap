//
//  LectureSnapApp.swift
//  LectureSnap
//
//  Created by Maxim Mikaya on 17.02.2025.
//

import SwiftUI
import AppKit

@main
struct LectureSnapApp: App {
    // Use an app delegate to manage the status item and panel
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var persistentPanel: PersistentPanel!

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create a status item in the menu bar
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "tray.fill", accessibilityDescription: "LectureSnap")
            button.action = #selector(togglePanel)
            button.target = self
        }
        
        // Create the persistent panel with your MenuBarDropView
        let hostingController = NSHostingController(rootView: MenuBarDropView())
        persistentPanel = PersistentPanel(
            contentRect: NSRect(x: 0, y: 0, width: 300, height: 200),
            styleMask: [.titled, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        persistentPanel.contentViewController = hostingController
        persistentPanel.isFloatingPanel = true
        // Set this to false to keep the panel visible even when the app loses focus.
        persistentPanel.hidesOnDeactivate = false
    }
    
    @objc func togglePanel() {
        if persistentPanel.isVisible {
            persistentPanel.orderOut(nil)
        } else {
            if let button = statusItem.button {
                // Position the panel relative to the status item
                let buttonFrame = button.window?.convertToScreen(button.frame) ?? NSRect.zero
                let panelOrigin = NSPoint(x: buttonFrame.minX, y: buttonFrame.minY - persistentPanel.frame.height)
                persistentPanel.setFrameOrigin(panelOrigin)
                persistentPanel.makeKeyAndOrderFront(nil)
            }
        }
    }
}
