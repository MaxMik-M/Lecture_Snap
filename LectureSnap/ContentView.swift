import SwiftUI
import PDFKit
import UniformTypeIdentifiers
import Vision
import Combine


// MARK: - Data Models

struct MainFolder: Identifiable, Equatable, Hashable {
    let id = UUID()
    var name: String
    var url: URL            // This is the resolved URL (for display)
    var bookmarkData: Data  // This is the persisted bookmark data
    var lectureFolders: [LectureFolder] = []
}

struct LectureFolder: Identifiable, Equatable, Hashable {
    let id = UUID()
    var name: String
    var url: URL
}





// MARK: - Loading Lecture Folders

func loadLectureFolders(from folder: URL) -> [LectureFolder] {
    // Must open security scope here for sandboxed apps
    guard folder.startAccessingSecurityScopedResource() else {
        print("Cannot access folder security scope.")
        return []
    }
    defer { folder.stopAccessingSecurityScopedResource() }

    do {
        let contents = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        let folders = contents.filter { $0.hasDirectoryPath }
        return folders.map { LectureFolder(name: $0.lastPathComponent, url: $0) }
    } catch {
        print("Error loading lecture folders: \(error)")
        return []
    }
}

// MARK: - OCR / Image Text Extraction

func extractTextFromImage(url: URL) -> String? {
    guard let ciImage = CIImage(contentsOf: url) else {
        print("Unable to create CIImage from file at \(url)")
        return nil
    }
    
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
    do {
        try handler.perform([request])
        guard let observations = request.results else { return nil }
        let text = observations.compactMap { observation in
            observation.topCandidates(1).first?.string
        }.joined(separator: "\n")
        return text.isEmpty ? nil : text
    } catch {
        print("OCR error: \(error)")
        return nil
    }
}

// MARK: - PDF Text Extraction

func extractTextFromPDF(url: URL, pageCount: Int) -> String? {
    guard let pdfDoc = PDFDocument(url: url) else { return nil }
    var extractedText = ""
    for i in 0..<min(pageCount, pdfDoc.pageCount) {
        if let page = pdfDoc.page(at: i), let text = page.string {
            extractedText += text + "\n\n"
        }
    }
    return extractedText.isEmpty ? nil : extractedText
}

// MARK: - Ollama Output Parsing

func extractCourseAndFolderFromOllamaOutput(_ rawOutput: String) -> (subject: String, courseFolder: String) {
    // Try to parse as JSON
    var contentText = rawOutput
    if let data = rawOutput.data(using: .utf8) {
        do {
            if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
               let choices = json["choices"] as? [[String: Any]],
               let message = choices.first?["message"] as? [String: Any],
               let content = message["content"] as? String {
                contentText = content
            }
        } catch {
            print("Failed to parse outer JSON: \(error)")
        }
    }
    
    // Cleanup
    var cleaned = contentText.trimmingCharacters(in: .whitespacesAndNewlines)
    cleaned = cleaned.replacingOccurrences(of: "<think>", with: "")
    cleaned = cleaned.replacingOccurrences(of: "</think>", with: "")
    if let finishRange = cleaned.range(of: "},\"finish_reason\"") {
        cleaned = String(cleaned[..<finishRange.lowerBound])
    }
    
    // Regex approach
    var subject = "Unknown Subject"
    var courseFolder = "No Matching Course Found"
    
    do {
        let subjectRegex = try NSRegularExpression(pattern: "[\\s\\n]*Subject:\\s*(.+)", options: .caseInsensitive)
        let courseRegex  = try NSRegularExpression(pattern: "[\\s\\n]*Course Folder:\\s*(.+)", options: .caseInsensitive)
        let nsCleaned = cleaned as NSString
        
        if let subjectMatch = subjectRegex.firstMatch(in: cleaned, options: [], range: NSRange(location: 0, length: nsCleaned.length)) {
            subject = nsCleaned.substring(with: subjectMatch.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let courseMatch = courseRegex.firstMatch(in: cleaned, options: [], range: NSRange(location: 0, length: nsCleaned.length)) {
            courseFolder = nsCleaned.substring(with: courseMatch.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
    } catch {
        print("Regex error: \(error)")
    }
    
    return (subject, courseFolder)
}

// MARK: - Fuzzy Matching (Levenshtein)

func levenshtein(_ aStr: String, _ bStr: String) -> Int {
    let a = Array(aStr.lowercased())
    let b = Array(bStr.lowercased())
    let empty = [Int](repeating: 0, count: b.count + 1)
    var last = [Int](0...b.count)
    for (i, ca) in a.enumerated() {
        var cur = [i + 1] + empty
        for (j, cb) in b.enumerated() {
            cur[j + 1] = ca == cb ? last[j] : min(last[j], last[j + 1], cur[j]) + 1
        }
        last = cur
    }
    return last.last!
}

func findClosestMatch(bestMatch: String, in folders: [LectureFolder]) -> LectureFolder? {
    var bestFolder: LectureFolder? = nil
    var smallestDistance = Int.max
    for folder in folders {
        let distance = levenshtein(folder.name, bestMatch)
        if distance < smallestDistance {
            smallestDistance = distance
            bestFolder = folder
        }
    }
    return smallestDistance <= 1 ? bestFolder : nil
}

// MARK: - Helper: Extract Both Course and Folder from LLM Output

func extractCourseAndFolder(from output: String) -> (subject: String, courseFolder: String) {
    let cleanedOutput = output
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .replacingOccurrences(of: "```json", with: "")
        .replacingOccurrences(of: "```", with: "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    
    if let data = cleanedOutput.data(using: .utf8) {
        do {
            if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: String] {
                let subject = json["subject"] ?? "Unknown Subject"
                let courseFolder = json["course_folder"] ?? "No Matching Course Found"
                return (subject, courseFolder)
            }
        } catch {
            print("JSON parsing error: \(error)")
        }
    }
    return ("Unknown Subject", "No Matching Course Found")
}

// MARK: - Model Source enum

enum ModelSource: String, CaseIterable, Identifiable {
    case api = "api"
    case local = "local"
    var id: String { rawValue }
}

/// Removes any `<think> ... </think>` blocks (including the text in between).
/// Also optionally removes extra newlines or markup if desired.
func removeChainOfThought(from text: String) -> String {
    // Regex matching `<think>` followed by any characters (including newlines),
    // up to `</think>`, in a DOTALL mode.
    let pattern = "<think>.*?</think>"
    guard let regex = try? NSRegularExpression(pattern: pattern,
                                               options: [.dotMatchesLineSeparators]) else {
        return text
    }
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    let stripped = regex.stringByReplacingMatches(in: text,
                                                  options: [],
                                                  range: range,
                                                  withTemplate: "")
    return stripped
}

func parseLocalLLMResponse(_ data: Data) -> String {
    // Convert raw data to dictionary
    guard let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
          let choices = json["choices"] as? [[String: Any]],
          let firstChoice = choices.first,
          let message = firstChoice["message"] as? [String: Any],
          let content = message["content"] as? String
    else {
        // If parsing fails, return the raw string or an error
        return String(data: data, encoding: .utf8) ?? "Error parsing JSON"
    }
    
    // Remove the chain-of-thought blocks
    let cleaned = removeChainOfThought(from: content)
    
    // If you also want to remove leftover tags or extra whitespace, do more cleanup:
    let final = cleaned
        .replacingOccurrences(of: "\n\n", with: "\n")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    
    return final
}



// MARK: - Run LLM

func runLLM(prompt: String,
            extractedText: String? = nil,
            lectureNames: [String]? = nil,
            needsClassification: Bool = true) -> String {
    let mode = UserDefaults.standard.string(forKey: "modelSource") ?? ModelSource.api.rawValue
    var finalPrompt = prompt

    if mode == ModelSource.local.rawValue{
        if needsClassification {
        // If local mode
            guard let text = extractedText, let lectures = lectureNames else {
                print("Missing extractedText or lectureNames for local mode")
                return "Error: Missing extracted data"
        }
        finalPrompt = """
        <|User|> Below is text extracted from a PDF.
        Please identify the academic course name of the text.
        Extracted Text:
        \(text)
        
        The available courses are:
        \(lectures.joined(separator: "\n"))
        Please choose the best matching course from the list above that fits the detected course.
        Return exactly the course name as it appears in the list, or return "No Matching Course Found" if none match.
        Return your answer with exactly two lines in the format:
        
        Subject:<Extracted course name>
        Course Folder:<Matching lecture folder name>
        
        <|Assistant|>
        """
    } else {
        // Summaries or other usage that doesn't need lectureNames
        finalPrompt = prompt
    }
}
    
    // Endpoint + model
    var endpoint: URL?
    let modelName: String
    if mode == ModelSource.api.rawValue {
        guard let apiKey = UserDefaults.standard.string(forKey: "openaiAPIKey"), !apiKey.isEmpty else {
            print("API key not set.")
            return "Error: API key not set."
        }
        endpoint = URL(string: "https://api.openai.com/v1/chat/completions")
        modelName = "gpt-4o-mini"  // your custom name
    } else {
        guard let server = UserDefaults.standard.string(forKey: "ollamaServer"), !server.isEmpty else {
            print("Ollama server not set.")
            return "Error: Ollama server not set."
        }
        endpoint = URL(string: "\(server)/v1/chat/completions")
        modelName = UserDefaults.standard.string(forKey: "ollamaModel") ?? "deepseek-r1:14b"
    }
    
    guard let url = endpoint else { return "Error: Invalid endpoint." }
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    if mode == ModelSource.api.rawValue {
        // pass Bearer token
        let apiKey = UserDefaults.standard.string(forKey: "openaiAPIKey")!
        request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    
    let messages: [[String: String]] = [
        ["role": "user", "content": finalPrompt]
    ]
    var payload: [String: Any] = [
        "model": modelName,
        "messages": messages,
        "temperature": 0.7
    ]
    if mode == ModelSource.api.rawValue {
        payload["max_tokens"] = 256
    }
    
    do {
        request.httpBody = try JSONSerialization.data(withJSONObject: payload, options: [])
    } catch {
        print("Error serializing JSON: \(error)")
        return "Error: Could not serialize JSON"
    }
    
    var result = ""
    let semaphore = DispatchSemaphore(value: 0)
    
    let task = URLSession.shared.dataTask(with: request) { data, response, error in
        defer { semaphore.signal() }
        
        if let error = error {
            print("Request error: \(error)")
            result = "Error: \(error.localizedDescription)"
            return
        }
        
        guard let data = data else {
            print("No data received.")
            result = "Error: No data received"
            return
        }
        
        if let rawString = String(data: data, encoding: .utf8) {
            print("Raw API response: \(rawString)")
        }
        
        if mode == ModelSource.api.rawValue {
            do {
                if let jsonResponse = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                   let choices = jsonResponse["choices"] as? [[String: Any]],
                   let message = choices.first?["message"] as? [String: Any],
                   let content = message["content"] as? String {
                    result = content.trimmingCharacters(in: .whitespacesAndNewlines)
                } else {
                    print("Unexpected response format.")
                    result = "Error: Unexpected response format"
                }
            } catch {
                print("Error parsing JSON: \(error)")
                result = "Error: \(error.localizedDescription)"
            }
        } else {
            if needsClassification {
                result = String(data: data, encoding: .utf8) ?? "Error: Could not decode response"}
            else {
                result = parseLocalLLMResponse(data)
            }
        }
    }
    task.resume()
    semaphore.wait()
    return result
}



func classifyCourse(for text: String, lectureNames: [String]) -> (subject: String, courseFolder: String) {
    let mode = UserDefaults.standard.string(forKey: "modelSource") ?? ModelSource.api.rawValue
    var prompt: String
    if mode == ModelSource.local.rawValue {
        prompt = """
        <|User|> Below is text extracted from a PDF.
        Please identify the academic course name of the text.
        Extracted Text:
        \(text)
        
        The available courses are:
        \(lectureNames.joined(separator: "\n"))
        
        Please choose the best matching course from the list above that fits the detected course.
        Return exactly the course name as it appears in the list, or return \"No Matching Course Found\" if none match.
        Return your answer with exactly two lines in the format:
        Subject:<Extracted course name>
        Course Folder:<Matching lecture folder name>
        <|Assistant|>
        """
        let rawOutput = runLLM(prompt: prompt, extractedText: text, lectureNames: lectureNames, needsClassification: true)
        return extractCourseAndFolderFromOllamaOutput(rawOutput)
    } else {
        prompt = """
        <|User|> Below is text extracted from a PDF.
        Please identify the academic course name of the text and choose the best matching course from the list.
        
        Extracted Text:
        \(text)
        
        The available courses are:
        \(lectureNames.joined(separator: "\n"))
        
        Return a JSON object with keys \"subject\" and \"course_folder\".
        Example: {\"subject\": \"Quantum Mechanics\", \"course_folder\": \"Quantum Computing\"}
        <|Assistant|>
        """
        let rawOutput = runLLM(prompt: prompt)
        return extractCourseAndFolder(from: rawOutput)
    }
}

// MARK: - SummaryDropView (Summary Mode)

struct SummaryDropView: View {
    @State private var instructionText: String = "Drop a PDF or image here to see its summary."
    @State private var isBusy = false
    
    // NEW STATES FOR THE SHEET
    @State private var summaryText: String = ""
    @State private var showingSummarySheet = false
    
    var body: some View {
        VStack {
            Text(instructionText)
                .padding()
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
        .contentShape(Rectangle())
        .onDrop(of: [UTType.fileURL.identifier], isTargeted: nil) { providers in
            guard !providers.isEmpty else { return false }
            // We’ll just process the first file for simplicity
            if let provider = providers.first {
                loadFileURL(from: provider) { url in
                    guard let url = url else { return }
                    processFileForSummary(url)
                }
            }
            return true
        }
        // PRESENT THE SUMMARY IN A SHEET
        .sheet(isPresented: $showingSummarySheet) {
            VStack(alignment: .leading) {
                Text("Summary")
                    .font(.title2)
                    .bold()
                    .padding(.bottom, 8)
                
                ScrollView {
                    Text(summaryText)
                        .textSelection(.enabled)
                        .padding(.bottom, 20)
                }
                
                Divider()
                
                HStack {
                    Spacer()
                    Button("Close") {
                        showingSummarySheet = false
                    }
                    .padding(.top, 8)
                }
            }
            .padding()
            .frame(minWidth: 400, minHeight: 300)
        }
    }
    
    /// Helper to load a URL from the NSItemProvider
    private func loadFileURL(from provider: NSItemProvider, completion: @escaping (URL?) -> Void) {
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { (item, error) in
            guard let data = item as? Data else {
                completion(nil)
                return
            }
            // Convert raw data to a file URL
            let url = NSURL(absoluteURLWithDataRepresentation: data, relativeTo: nil) as URL
            completion(url)
        }
    }
    
    private func processFileForSummary(_ fileURL: URL) {
        isBusy = true
        instructionText = "Extracting text from \(fileURL.lastPathComponent)..."
        
        DispatchQueue.global(qos: .userInitiated).async {
            let ext = fileURL.pathExtension.lowercased()
            var extractedText: String? = nil
            
            if ext == "pdf" {
                extractedText = extractTextFromPDF(url: fileURL, pageCount: 2)
            } else if ["jpg", "jpeg", "png", "tiff"].contains(ext) {
                extractedText = extractTextFromImage(url: fileURL)
            } else {
                extractedText = nil
            }
            
            guard let text = extractedText, !text.isEmpty else {
                DispatchQueue.main.async {
                    instructionText = "Could not extract text from \(fileURL.lastPathComponent)."
                    isBusy = false
                }
                return
            }
            
            // Now call LLM to get a summary
            let prompt = """
            <|User|> Summarize the following text in a short paragraph with 3-5 sentences.

            \(text)

            <|Assistant|>
            """
            let rawSummary = runLLM(prompt: prompt, needsClassification: false)
            let cleanedSummary = rawSummary.trimmingCharacters(in: .whitespacesAndNewlines)
            
            DispatchQueue.main.async {
                summaryText = """
                Summary of \(fileURL.lastPathComponent):
                \(cleanedSummary)
                """
                showingSummarySheet = true
                instructionText = "Drop another file to see a summary."
                isBusy = false
            }
        }
    }
}

// MARK: - Main ContentView

struct ContentView: View {
    @State private var outputText = "Drag & drop a PDF or an image to sort"
    @State private var detectedSubject: String = ""
    @State private var matchedLecture: String = ""
    @State private var isShowingResult = false
    @State private var showingSettings = false
    
    // NEW: Toggle to switch between Sorting and Summary
    @State private var isSummaryMode = false
    
    // API key (stored in UserDefaults as needed)
    @AppStorage("openaiAPIKey") var openaiAPIKey: String = ""
    
    // Folders for “destination courses”
    @State private var mainFolders: [MainFolder] = []
    @State private var selectedMainFolder: MainFolder? = nil
    
    // The array of possible “source folder” bookmarks
    @State private var sourceFolderBookmarks: [Data] = []
    
    var body: some View {
        NavigationView {
            // --- Sidebar on the left ---
            SidebarView(
                mainFolders: $mainFolders,
                selectedMainFolder: $selectedMainFolder
            )
            
            // --- Main content area on the right ---
            VStack {
                if isSummaryMode {
                    // ---- SUMMARY MODE ----
                    Text("Summary Mode: Drop a PDF or image to get a short summary.")
                        .padding()
                    SummaryDropView()
                        .frame(width: 300, height: 200)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(12)
                        .padding()
                } else {
                    // ---- SORTING MODE (existing) ----
                    Text(outputText)
                        .padding()
                        .multilineTextAlignment(.center)
                    
                    if let _ = selectedMainFolder {
                        DropView(
                            outputText: $outputText,
                            detectedSubject: $detectedSubject,
                            matchedLecture: $matchedLecture,
                            isShowingResult: $isShowingResult,
                            lectureNames: bindingForSelectedLectureNames(),
                            selectedMainFolder: $selectedMainFolder,
                            sourceFolderBookmarks: $sourceFolderBookmarks
                        )
                        .frame(width: 300, height: 200)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(12)
                        .padding()
                    } else {
                        Text("Select a main folder from the sidebar")
                            .foregroundColor(.gray)
                            .padding()
                    }
                }
                
                Spacer()
                
                // Toggle can go here or in a toolbar
                Toggle(isOn: $isSummaryMode) {
                    Text("Summary Mode")
                }
                .padding()
                .toggleStyle(.switch)
            }
            .frame(minWidth: 600, minHeight: 400)
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    // Button to open settings
                    Button(action: { showingSettings = true }) {
                        Image(systemName: "gear")
                    }
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView(
                    apiKey: $openaiAPIKey,
                    sourceFolderBookmarks: $sourceFolderBookmarks
                )
            }
            .sheet(isPresented: $isShowingResult) {
                // The result sheet for Sorting Mode
                VStack {
                    Text("Detected Subject & Lecture")
                        .font(.headline)
                        .padding()
                    ScrollView {
                        Text("Subject: \(detectedSubject)")
                            .font(.title2)
                            .bold()
                            .padding()
                        Text("Course: \(matchedLecture)")
                            .font(.title2)
                            .bold()
                            .foregroundColor(.blue)
                            .padding()
                    }
                    Button("Close") {
                        isShowingResult = false
                    }
                    .padding()
                }
                .frame(minWidth: 320, minHeight: 180)
            }
        }
        .onAppear {
            // Load main folders from UserDefaults
            mainFolders = loadMainFolders()
            // Also load any previously saved source folders
            sourceFolderBookmarks = loadSourceFolderBookmarks()
        }
        .onChange(of: mainFolders) { _, newFolders in
            saveMainFolders(newFolders)
        }
    }
    
    // Helper: Binding for lecture names in the selected folder
    func bindingForSelectedLectureNames() -> Binding<[String]> {
        Binding<[String]>(
            get: {
                if let selectedId = selectedMainFolder?.id,
                   let folder = mainFolders.first(where: { $0.id == selectedId }) {
                    return folder.lectureFolders.map { $0.name }
                }
                return []
            },
            set: { newNames in
                if let selectedId = selectedMainFolder?.id,
                   let index = mainFolders.firstIndex(where: { $0.id == selectedId }) {
                    let oldLectures = mainFolders[index].lectureFolders
                    let updatedLectures = zip(oldLectures, newNames).map { (old, new) in
                        LectureFolder(name: new, url: old.url)
                    }
                    mainFolders[index].lectureFolders = updatedLectures
                    selectedMainFolder = mainFolders[index]
                }
            }
        )
    }
}
// MARK: - SettingsView

struct SettingsView: View {
    // Existing API-related properties
    @Binding var apiKey: String
    @AppStorage("modelSource") var modelSource: String = ModelSource.api.rawValue
    @AppStorage("ollamaServer") var ollamaServer: String = "http://localhost:11434"
    @AppStorage("ollamaModel") var ollamaModel: String = "deepseek-r1:14b"
    
    @Environment(\.presentationMode) var presentationMode
    
    // NEW: The array of source folder bookmarks
    @Binding var sourceFolderBookmarks: [Data]
    
    // A computed property to show each saved folder in the list
    var resolvedSourceFolders: [URL] {
        sourceFolderBookmarks.compactMap { data in
            var isStale = false
            if let url = try? URL(resolvingBookmarkData: data,
                                  options: .withSecurityScope,
                                  relativeTo: nil,
                                  bookmarkDataIsStale: &isStale),
               !isStale {
                return url
            }
            return nil
        }
    }
    
    var body: some View {
        ZStack {
            Color.clear // Make the background transparent
                .ignoresSafeArea()
            VStack {
                TabView {
                    // ---------------------
                    // Tab 1: LLM Settings
                    // ---------------------
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Model Source")
                            .font(.title2)
                            .bold()
                        
                        Picker("Model Source", selection: $modelSource) {
                            ForEach(ModelSource.allCases) { source in
                                Text(source == .api ? "Remote API" : "Local Ollama")
                                    .tag(source.rawValue)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        

                        if modelSource == ModelSource.api.rawValue {
                            Text("Enter your OpenAI API Key below:")
                                .font(.subheadline)
                            TextField("API Key", text: $apiKey)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .textContentType(.password)
                                .disableAutocorrection(true)
                        } else {
                            Text("Enter your Ollama Server URL below:")
                                .font(.subheadline)
                            TextField("Ollama Server", text: $ollamaServer)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .disableAutocorrection(true)
                            
                            Text("Enter your Ollama Model Name:")
                                .font(.subheadline)
                            TextField("Ollama Model", text: $ollamaModel)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .disableAutocorrection(true)
                            
                            Text("Note: Recommended to use deepseek-r1\nwith 7b or more parameters ")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        
                        Spacer()
                        HStack {
                                        Spacer() // Pushes the button to the right
                                        Button("Done") {
                                            presentationMode.wrappedValue.dismiss()
                                        }
                                    }
                    }
                    .padding(24)
                    .tabItem {
                        Label("LLM Settings", systemImage: "gearshape")
                    }
                    
                    // ---------------------
                    // Tab 2: Source Folders
                    // ---------------------
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Pick the folder(s) from which you regularly move files:")
                            .font(.title2)
                            .bold()
                        
                        List {
                            ForEach(resolvedSourceFolders, id: \.self) { folderURL in
                                HStack {
                                    Text(folderURL.lastPathComponent)
                                    Spacer()
                                    Button(action: {
                                        removeSourceFolder(folderURL)
                                    }) {
                                        Image(systemName: "trash")
                                            .foregroundColor(.red)
                                    }
                                    .buttonStyle(BorderlessButtonStyle())
                                }
                            }
                        }
                        .frame(minHeight: 150)
                        
                        Button("Add More Folders") {
                            pickMoreFolders()
                        }
                        .padding(.top, 8)
                        
                        Spacer()
                        HStack {
                                        Spacer() // Pushes the button to the right
                                        Button("Done") {
                                            presentationMode.wrappedValue.dismiss()
                                        }
                                    }
                    }
                    .padding(24)
                    .tabItem {
                        Label("Source Folders", systemImage: "folder")
                    }
                }
                .padding()
                .background(Color(NSColor.windowBackgroundColor))
                .cornerRadius(10)
                .navigationTitle("Settings")

            }
        }
    }
    
    // MARK: - Add More Folders
    func pickMoreFolders() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            for folderURL in panel.urls {
                do {
                    let bookmarkData = try folderURL.bookmarkData(
                        options: .withSecurityScope,
                        includingResourceValuesForKeys: nil,
                        relativeTo: nil
                    )
                    sourceFolderBookmarks.append(bookmarkData)
                } catch {
                    print("Error creating bookmark for source folder: \(error)")
                }
            }
            // Save them so we have them on next launch
            saveSourceFolderBookmarks(sourceFolderBookmarks)
        }
    }
    
    // MARK: - Remove a folder from sourceFolderBookmarks
    func removeSourceFolder(_ folderURL: URL) {
        if let index = sourceFolderBookmarks.firstIndex(where: { data in
            var isStale = false
            if let candidate = try? URL(resolvingBookmarkData: data,
                                        options: .withSecurityScope,
                                        relativeTo: nil,
                                        bookmarkDataIsStale: &isStale),
               !isStale,
               candidate.path == folderURL.path {
                return true
            }
            return false
        }) {
            sourceFolderBookmarks.remove(at: index)
            saveSourceFolderBookmarks(sourceFolderBookmarks)
        }
    }
}

// MARK: - Helper: Resolve Main Folder URL from Bookmark

func resolveMainFolderURL(from folder: MainFolder) -> URL? {
    var isStale = false
    if let url = try? URL(resolvingBookmarkData: folder.bookmarkData,
                          options: .withSecurityScope,
                          relativeTo: nil,
                          bookmarkDataIsStale: &isStale) {
        if isStale {
            print("Bookmark data is stale. Please re-select the folder.")
            return nil
        }
        return url
    }
    return nil
}

// MARK: - SidebarView

struct SidebarView: View {
    @Binding var mainFolders: [MainFolder]
    @Binding var selectedMainFolder: MainFolder?
    
    @State private var isEditing = false

    var body: some View {
        VStack {
            HStack {
                Button("Add Folder with Courses") {
                    addMainFolder()
                }
                .padding(.leading)
                Button(isEditing ? "Done" : "Edit") {
                    isEditing.toggle()
                }
                .padding(.trailing)
            }
            List {
                ForEach($mainFolders) { $folder in
                    DisclosureGroup(isExpanded: Binding(
                        get: { isEditing || (selectedMainFolder?.id == folder.id) },
                        set: { expanded in
                            if !isEditing {
                                if expanded { selectedMainFolder = folder } else { selectedMainFolder = nil }
                            }
                        }
                    )) {
                        ForEach($folder.lectureFolders) { $lecture in
                            HStack {
                                if isEditing {
                                    TextField("Lecture Name", text: $lecture.name, onCommit: {
                                        renameLecture(folder: folder, lecture: lecture)
                                    })
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                } else {
                                    Text(lecture.name)
                                }
                            }
                        }
                        if selectedMainFolder?.id == folder.id {
                            Button("Add a Course") {
                                addLecture(to: folder)
                            }
                            .padding(.leading, 20)
                        }
                    } label: {
                        HStack {
                            Text(folder.name)
                                .padding(4)
                                .background(selectedMainFolder?.id == folder.id ? Color.blue.opacity(0.2) : Color.clear)
                                .cornerRadius(4)
                                .onTapGesture(count: 2) {
                                    selectedMainFolder = folder
                                }
                            if isEditing {
                                Button(action: {
                                    removeMainFolder(folder: folder)
                                }) {
                                    Image(systemName: "minus.circle.fill")
                                        .foregroundColor(.red)
                                }
                                .buttonStyle(BorderlessButtonStyle())
                            }
                        }
                    }
                }
                .onDelete { indexSet in
                    mainFolders.remove(atOffsets: indexSet)
                    if let first = mainFolders.first {
                        selectedMainFolder = first
                    } else {
                        selectedMainFolder = nil
                    }
                }
            }
            .listStyle(SidebarListStyle())
        }
        .frame(minWidth: 250)
        .navigationTitle("Folders & Lectures")
    }
    
    func addMainFolder() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        if panel.runModal() == .OK, let folderURL = panel.urls.first {
            var isStale = false
            guard let bookmarkData = try? folderURL.bookmarkData(options: .withSecurityScope,
                                                                 includingResourceValuesForKeys: nil,
                                                                 relativeTo: nil) else { return }
            let resolvedURL = try? URL(resolvingBookmarkData: bookmarkData,
                                       options: .withSecurityScope,
                                       relativeTo: nil,
                                       bookmarkDataIsStale: &isStale)
            let newFolder = MainFolder(name: folderURL.lastPathComponent,
                                       url: resolvedURL ?? folderURL,
                                       bookmarkData: bookmarkData,
                                       lectureFolders: loadLectureFolders(from: folderURL))
            mainFolders.append(newFolder)
            selectedMainFolder = newFolder
        }
    }
    
    func removeMainFolder(folder: MainFolder) {
        if let index = mainFolders.firstIndex(where: { $0.id == folder.id }) {
            mainFolders.remove(at: index)
            if selectedMainFolder?.id == folder.id {
                selectedMainFolder = mainFolders.first
            }
        }
    }
    
    func renameLecture(folder: MainFolder, lecture: LectureFolder) {
        let parentURL = lecture.url.deletingLastPathComponent()
        let newURL = parentURL.appendingPathComponent(lecture.name)
        do {
            try FileManager.default.moveItem(at: lecture.url, to: newURL)
            if let folderIndex = mainFolders.firstIndex(where: { $0.id == folder.id }),
               let lectureIndex = mainFolders[folderIndex].lectureFolders.firstIndex(where: { $0.id == lecture.id }) {
                mainFolders[folderIndex].lectureFolders[lectureIndex].url = newURL
            }
        } catch {
            print("Error renaming lecture folder: \(error)")
        }
    }
    
    func addLecture(to folder: MainFolder) {
        let newLectureName = "New Lecture \(folder.lectureFolders.count + 1)"
        let newLectureURL = folder.url.appendingPathComponent(newLectureName)
        do {
            try FileManager.default.createDirectory(at: newLectureURL, withIntermediateDirectories: true, attributes: nil)
            if let index = mainFolders.firstIndex(where: { $0.id == folder.id }) {
                mainFolders[index].lectureFolders.append(LectureFolder(name: newLectureName, url: newLectureURL))
                selectedMainFolder = mainFolders[index]
            }
        } catch {
            print("Error creating lecture folder: \(error)")
        }
    }
}

// MARK: - DropView

struct DropView: View {
    @Binding var outputText: String
    @Binding var detectedSubject: String
    @Binding var matchedLecture: String
    @Binding var isShowingResult: Bool
    @Binding var lectureNames: [String]
    @Binding var selectedMainFolder: MainFolder?
    
    @State private var fileResults: [String] = []
    
    // NEW: List of source folder bookmarks
    @Binding var sourceFolderBookmarks: [Data]
    
    var body: some View {
        VStack {
            Text("Drop your files here")
                .foregroundColor(.gray)
                .padding()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .cornerRadius(12)
        .contentShape(Rectangle())
        .onDrop(of: [UTType.fileURL.identifier], isTargeted: nil) { providers in
            guard !providers.isEmpty else { return false }
            DispatchQueue.main.async {
                outputText = "Processing file(s)..."
                fileResults.removeAll()
                processNextFile(providers, index: 0)
            }
            return true
        }
        
    }
    
    // Find which source folder (if any) contains the dropped file
    func findSourceFolderFor(fileURL: URL) -> URL? {
        let filePath = fileURL.path
        for bookmarkData in sourceFolderBookmarks {
            var isStale = false
            do {
                let folderURL = try URL(resolvingBookmarkData: bookmarkData,
                                        options: .withSecurityScope,
                                        relativeTo: nil,
                                        bookmarkDataIsStale: &isStale)
                if isStale { continue }
                // Check if file is inside folderURL
                if filePath.hasPrefix(folderURL.path) {
                    return folderURL
                }
            } catch {
                print("Error resolving source folder bookmark: \(error)")
            }
        }
        return nil
    }
    
    func processNextFile(_ providers: [NSItemProvider], index: Int) {
        if index >= providers.count {
            DispatchQueue.main.async {
                let combined = fileResults.joined(separator: "\n\n")
                detectedSubject = "File(s) Processed"
                matchedLecture = combined
                isShowingResult = true
                outputText = "Finished analyzing \(providers.count) file(s)."
            }
            return
        }
        
        let provider = providers[index]
        _ = provider.loadObject(ofClass: URL.self) { (url, error) in
            guard let fileURL = url else {
                // If we can't get a URL, move on to the next file
                processNextFile(providers, index: index + 1)
                return
            }
            
            // Update UI on main thread
            DispatchQueue.main.async {
                outputText = "Processing file \(index + 1) of \(providers.count): \(fileURL.lastPathComponent)"
            }
            
            // Heavy work on background thread
            DispatchQueue.global(qos: .userInitiated).async {
                // Extract text from PDF or image
                let ext = fileURL.pathExtension.lowercased()
                var extractedText: String?
                
                if ext == "pdf" {
                    extractedText = extractTextFromPDF(url: fileURL, pageCount: 2)
                } else if ["jpg", "jpeg", "png", "tiff"].contains(ext) {
                    extractedText = extractTextFromImage(url: fileURL)
                } else {
                    print("Unsupported file format: \(ext)")
                }
                
                // If we have extracted text
                if let text = extractedText {
                    
                    let (subject, courseFolder) = classifyCourse(for: text, lectureNames: lectureNames)
                    
                    // Update SwiftUI state on the main thread
                    DispatchQueue.main.async {
                        detectedSubject = subject
                        matchedLecture = courseFolder
                    }
                    
                    // 2) Reload lecture folders from disk
                    if var updatedMainFolder = selectedMainFolder {
                        updatedMainFolder.lectureFolders = loadLectureFolders(from: updatedMainFolder.url)
                        selectedMainFolder = updatedMainFolder
                    }
                    
                    // 3) If LLM says "No Matching Course Found", skip moving the file
                    if courseFolder == "No Matching Course Found" {
                        print("LLM indicated no matching folder — skipping file move.")
                    } else {
                        // Otherwise, try to move the file
                        if let updatedMainFolder = selectedMainFolder {
                            let exactMatch = updatedMainFolder.lectureFolders.first {
                                $0.name.lowercased() == courseFolder.lowercased()
                            }
                            let targetLectureFolder = exactMatch
                            ?? findClosestMatch(bestMatch: courseFolder, in: updatedMainFolder.lectureFolders)
                            
                            if let lectureFolder = targetLectureFolder {
                                // Check if file is in a known source folder
                                if let sourceFolderURL = findSourceFolderFor(fileURL: fileURL) {
                                    let sourceAccess = sourceFolderURL.startAccessingSecurityScopedResource()
                                    guard sourceAccess else {
                                        print("Could not start security scope for source folder: \(sourceFolderURL)")
                                        processNextFile(providers, index: index + 1)
                                        return
                                    }
                                    defer { sourceFolderURL.stopAccessingSecurityScopedResource() }
                                    
                                    // Also start security scope on the main folder
                                    if let resolvedURL = resolveMainFolderURL(from: updatedMainFolder) {
                                        let mainAccess = resolvedURL.startAccessingSecurityScopedResource()
                                        guard mainAccess else {
                                            print("Error: Could not access security scope for main folder.")
                                            processNextFile(providers, index: index + 1)
                                            return
                                        }
                                        defer { resolvedURL.stopAccessingSecurityScopedResource() }
                                        
                                        // Ensure lecture folder exists
                                        if !FileManager.default.fileExists(atPath: lectureFolder.url.path) {
                                            do {
                                                try FileManager.default.createDirectory(
                                                    at: lectureFolder.url,
                                                    withIntermediateDirectories: true,
                                                    attributes: nil
                                                )
                                            } catch {
                                                print("Error creating directory: \(error)")
                                            }
                                        }
                                        
                                        let destinationURL = lectureFolder.url.appendingPathComponent(fileURL.lastPathComponent)
                                        
                                        do {
                                            try FileManager.default.moveItem(at: fileURL, to: destinationURL)
                                            print("Moved file to: \(destinationURL)")
                                        } catch {
                                            print("Error moving file: \(error)")
                                        }
                                    } else {
                                        print("Failed to resolve main folder URL. Please re-select the folder.")
                                    }
                                } else {
                                    // Not from a known source folder, lacking permission
                                    print("File is not from a known source folder. No move performed.")
                                }
                            } else {
                                print("No matching lecture folder found for: \(courseFolder). File not moved.")
                            }
                        } else {
                            print("No main folder selected")
                        }
                    }
                    
                    // 4) Add result string, then queue next file
                    let resultString = """
                    File: \(fileURL.lastPathComponent)
                    Detected Course: \(subject)
                    Course Folder: \(courseFolder)
                    """
                    DispatchQueue.main.async {
                        fileResults.append(resultString)
                        processNextFile(providers, index: index + 1)
                    }
                    
                } else {
                    // extractedText == nil
                    DispatchQueue.main.async {
                        fileResults.append(
                            "File: \(fileURL.lastPathComponent)\nError: Failed to extract text."
                        )
                        processNextFile(providers, index: index + 1)
                    }
                }
            }
        }
    }
}

class PersistentPanel: NSPanel {
    override var canBecomeKey: Bool {
        return true
    }
}


// MARK: - Menu Bar View

struct MenuBarDropView: View {
    @State private var outputText: String = "Drop your files here"
    @State private var fileResults: [String] = []
    @State private var detectedSubject: String = ""
    @State private var matchedLecture: String = ""
    
    // Use main folders loaded from persistent storage
    @State private var mainFolders: [MainFolder] = loadMainFolders()
    @State private var selectedMainFolder: MainFolder? = nil
    // Lecture names for classification, derived from the selected folder
    @State private var lectureNames: [String] = []
    @State private var sourceFolderBookmarks: [Data] = loadSourceFolderBookmarks()
    
    var body: some View {
        VStack(alignment: .leading) {
            // Folder selector panel similar to main window
            HStack {
                Picker("Sort to Folder:", selection: $selectedMainFolder) {
                    ForEach(mainFolders, id: \.id) { folder in
                        Text(folder.name).tag(Optional(folder))
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }
            .padding([.top, .horizontal])
            
            Divider()
                .padding(.horizontal)
            
            Text(outputText)
                .foregroundColor(.gray)
                .padding()
            
            ScrollView {
                ForEach(fileResults, id: \.self) { result in
                    Text(result)
                        .padding(2)
                        .font(.system(size: 12))
                }
            }
            .padding(.horizontal)
        }
        .frame(width: 250, height: 200)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
        .contentShape(Rectangle())
        .onDrop(of: [UTType.fileURL.identifier], isTargeted: nil) { providers in
            guard !providers.isEmpty else { return false }
            outputText = "Processing \(providers.count) file(s)..."
            fileResults.removeAll()
            processNextFile(providers, index: 0)
            return true
        }
        .onAppear {
            mainFolders = loadMainFolders()
            if let first = mainFolders.first {
                selectedMainFolder = first
                lectureNames = first.lectureFolders.map { $0.name }
            }
        }
    }
    
    // Recursively process dropped files using the same logic as in your main DropView.
    func processNextFile(_ providers: [NSItemProvider], index: Int) {
        if index >= providers.count {
            DispatchQueue.main.async {
                outputText = "Finished processing \(providers.count) file(s)."
            }
            return
        }
        
        let provider = providers[index]
        _ = provider.loadObject(ofClass: URL.self) { (url, error) in
            guard let fileURL = url else {
                DispatchQueue.main.async {
                    processNextFile(providers, index: index + 1)
                }
                return
            }
            
            DispatchQueue.main.async {
                outputText = "Processing file \(index+1) of \(providers.count): \(fileURL.lastPathComponent)"
            }
            
            DispatchQueue.global(qos: .userInitiated).async {
                let ext = fileURL.pathExtension.lowercased()
                var extractedText: String?
                if ext == "pdf" {
                    extractedText = extractTextFromPDF(url: fileURL, pageCount: 2)
                } else if ["jpg", "jpeg", "png", "tiff"].contains(ext) {
                    extractedText = extractTextFromImage(url: fileURL)
                }
                
                if let text = extractedText {
                    // Use the unified classification function.
                    let (subject, courseFolder) = classifyCourse(for: text, lectureNames: lectureNames)
                    
                    DispatchQueue.main.async {
                        detectedSubject = subject
                        matchedLecture = courseFolder
                    }
                    
                    // Optionally, update your folder list and move the file as in your DropView.
                    if courseFolder != "No Matching Course Found",
                       let mainFolder = selectedMainFolder,
                       let exactMatch = mainFolder.lectureFolders.first(where: { $0.name.lowercased() == courseFolder.lowercased() })
                        ?? findClosestMatch(bestMatch: courseFolder, in: mainFolder.lectureFolders) {
                        
                        if let sourceFolderURL = findSourceFolderFor(fileURL: fileURL) {
                            let sourceAccess = sourceFolderURL.startAccessingSecurityScopedResource()
                            guard sourceAccess else {
                                DispatchQueue.main.async {
                                    fileResults.append("Error accessing source folder for \(fileURL.lastPathComponent)")
                                    processNextFile(providers, index: index+1)
                                }
                                return
                            }
                            defer { sourceFolderURL.stopAccessingSecurityScopedResource() }
                            
                            if let resolvedURL = resolveMainFolderURL(from: mainFolder) {
                                let mainAccess = resolvedURL.startAccessingSecurityScopedResource()
                                guard mainAccess else {
                                    DispatchQueue.main.async {
                                        fileResults.append("Error accessing main folder for \(fileURL.lastPathComponent)")
                                        processNextFile(providers, index: index+1)
                                    }
                                    return
                                }
                                defer { resolvedURL.stopAccessingSecurityScopedResource() }
                                
                                if !FileManager.default.fileExists(atPath: exactMatch.url.path) {
                                    try? FileManager.default.createDirectory(at: exactMatch.url, withIntermediateDirectories: true, attributes: nil)
                                }
                                let destinationURL = exactMatch.url.appendingPathComponent(fileURL.lastPathComponent)
                                try? FileManager.default.moveItem(at: fileURL, to: destinationURL)
                            }
                        }
                    }
                    
                    let resultString = """
                    File: \(fileURL.lastPathComponent)
                    Subject: \(subject)
                    Course Folder: \(courseFolder)
                    """
                    DispatchQueue.main.async {
                        fileResults.append(resultString)
                        processNextFile(providers, index: index+1)
                    }
                } else {
                    DispatchQueue.main.async {
                        fileResults.append("Failed to extract text from \(fileURL.lastPathComponent)")
                        processNextFile(providers, index: index+1)
                    }
                }
            }
        }
    }
    
    func findSourceFolderFor(fileURL: URL) -> URL? {
        let filePath = fileURL.path
        for bookmarkData in sourceFolderBookmarks {
            var isStale = false
            if let folderURL = try? URL(resolvingBookmarkData: bookmarkData,
                                        options: .withSecurityScope,
                                        relativeTo: nil,
                                        bookmarkDataIsStale: &isStale),
               !isStale,
               filePath.hasPrefix(folderURL.path) {
                return folderURL
            }
        }
        return nil
    }
}


// MARK: - Persistence Helpers: Main Folders

func saveMainFolders(_ folders: [MainFolder]) {
    var folderDicts: [[String: Any]] = []
    for folder in folders {
        let bookmarkData = folder.bookmarkData
        
        var lecturesArray: [[String: Any]] = []
        for lecture in folder.lectureFolders {
            lecturesArray.append(["name": lecture.name])
        }
        folderDicts.append([
            "name": folder.name,
            "urlBookmark": bookmarkData,
            "lectureFolders": lecturesArray
        ])
    }
    UserDefaults.standard.set(folderDicts, forKey: "SavedMainFolders")
}

func loadMainFolders() -> [MainFolder] {
    guard let folderDicts = UserDefaults.standard.array(forKey: "SavedMainFolders") as? [[String: Any]] else {
        return []
    }
    var folders: [MainFolder] = []
    for folderDict in folderDicts {
        guard let name = folderDict["name"] as? String,
              let bookmarkData = folderDict["urlBookmark"] as? Data else { continue }
        
        var isStale = false
        guard let url = try? URL(resolvingBookmarkData: bookmarkData,
                                 options: .withSecurityScope,
                                 relativeTo: nil,
                                 bookmarkDataIsStale: &isStale) else { continue }
        
        var lectures: [LectureFolder] = []
        if let lecturesArray = folderDict["lectureFolders"] as? [[String: Any]] {
            for lectureDict in lecturesArray {
                guard let lectureName = lectureDict["name"] as? String else { continue }
                let lectureURL = url.appendingPathComponent(lectureName)
                lectures.append(LectureFolder(name: lectureName, url: lectureURL))
            }
        }
        let mainFolder = MainFolder(name: name, url: url, bookmarkData: bookmarkData, lectureFolders: lectures)
        folders.append(mainFolder)
    }
    return folders
}

// MARK: - Persistence Helpers: Source Folders

func saveSourceFolderBookmarks(_ bookmarks: [Data]) {
    UserDefaults.standard.set(bookmarks, forKey: "SavedSourceFolderBookmarks")
}

func loadSourceFolderBookmarks() -> [Data] {
    guard let arr = UserDefaults.standard.array(forKey: "SavedSourceFolderBookmarks") as? [Data] else {
        return []
    }
    return arr
}


// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
