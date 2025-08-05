import React, { useState, useEffect } from 'react';
import './App.css';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import { AlertCircle, FileText, Search, Brain, CheckCircle2, Clock, Copy, Sparkles, Zap, Moon, Sun, ArrowRight, Download, Share2, BookOpen, Target, Lightbulb } from 'lucide-react';

function App() {
  const [documentUrl, setDocumentUrl] = useState('');
  const [questions, setQuestions] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults(null);

    try {
      const questionsList = questions.split('\n').filter(q => q.trim()).map(q => q.trim());
      
      if (questionsList.length === 0) {
        throw new Error('Please enter at least one question');
      }

      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/hackrx/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer 281434c7d6cd10c8ebce3a707b6166f84e6afbfd041a74f9f32c3cfe2c84fb01'
        },
        body: JSON.stringify({
          documents: documentUrl,
          questions: questionsList
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process request');
      }

      const data = await response.json();
      setResults({
        questions: questionsList,
        answers: data.answers
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadDemo = () => {
    setDocumentUrl('https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D');
    setQuestions(`What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
What is the waiting period for pre-existing diseases (PED) to be covered?
Does this policy cover maternity expenses, and what are the conditions?
What is the waiting period for cataract surgery?
Are the medical expenses for an organ donor covered under this policy?
What is the No Claim Discount (NCD) offered in this policy?
Is there a benefit for preventive health check-ups?
How does the policy define a 'Hospital'?
What is the extent of coverage for AYUSH treatments?
Are there any sub-limits on room rent and ICU charges for Plan A?`);
  };

  const copyToClipboard = async (text, index) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <div className={`min-h-screen transition-all duration-500 ${darkMode ? 'dark bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900' : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100'}`}>
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-pink-400/20 to-orange-600/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-cyan-400/10 to-blue-600/10 rounded-full blur-3xl animate-pulse delay-500"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 backdrop-blur-xl bg-white/70 dark:bg-slate-900/70 border-b border-white/20 dark:border-slate-700/50 shadow-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="p-3 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-2xl shadow-lg animate-pulse">
                  <Brain className="h-8 w-8 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full animate-ping"></div>
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                  HackRx AI
                </h1>
                <p className="text-slate-600 dark:text-slate-400 text-sm font-medium">
                  Intelligent Document Analysis Platform
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button
                onClick={() => setDarkMode(!darkMode)}
                variant="ghost"
                size="sm"
                className="rounded-full p-2 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all duration-200"
              >
                {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 container mx-auto px-6 py-8">
        <div className="max-w-5xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-blue-500/10 to-purple-500/10 dark:from-blue-400/20 dark:to-purple-400/20 px-4 py-2 rounded-full mb-6">
              <Sparkles className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Powered by Advanced AI</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-slate-800 dark:text-white mb-4">
              Transform Your Documents into
              <span className="block bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                Intelligent Insights
              </span>
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed">
              Upload any document and ask questions. Our AI will analyze the content and provide accurate, contextual answers instantly.
            </p>
          </div>

          {/* Main Form Card */}
          <Card className="backdrop-blur-xl bg-white/80 dark:bg-slate-800/80 border-white/20 dark:border-slate-700/50 shadow-2xl mb-8">
            <CardHeader className="pb-6">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-4 md:space-y-0">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                    <Target className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-2xl font-bold text-slate-800 dark:text-white">
                      Document Analysis
                    </CardTitle>
                    <CardDescription className="text-slate-600 dark:text-slate-400">
                      Get instant answers from your documents using AI
                    </CardDescription>
                  </div>
                </div>
                <Button 
                  onClick={loadDemo} 
                  variant="outline" 
                  className="shrink-0 border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all duration-200"
                >
                  <BookOpen className="h-4 w-4 mr-2" />
                  Load Demo
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-8">
                <div className="space-y-4">
                  <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
                    <FileText className="h-4 w-4 inline mr-2" />
                    Document URL
                  </label>
                  <Input
                    type="url"
                    value={documentUrl}
                    onChange={(e) => setDocumentUrl(e.target.value)}
                    placeholder="https://example.com/document.pdf"
                    required
                    className="w-full p-4 text-lg border-slate-200 dark:border-slate-600 bg-white/50 dark:bg-slate-700/50 backdrop-blur-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  />
                </div>

                <div className="space-y-4">
                  <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
                    <Lightbulb className="h-4 w-4 inline mr-2" />
                    Questions (one per line)
                  </label>
                  <Textarea
                    value={questions}
                    onChange={(e) => setQuestions(e.target.value)}
                    placeholder={`Enter your questions, one per line:
What is the main topic of this document?
What are the key findings?
What recommendations are provided?`}
                    required
                    rows={8}
                    className="w-full p-4 text-lg border-slate-200 dark:border-slate-600 bg-white/50 dark:bg-slate-700/50 backdrop-blur-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
                  />
                </div>

                <Button 
                  type="submit" 
                  disabled={loading}
                  className="w-full py-4 text-lg font-semibold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 text-white border-none shadow-lg transition-all duration-300 hover:shadow-xl hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {loading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                      <span>Processing Document...</span>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center">
                      <Zap className="h-5 w-5 mr-3" />
                      Analyze Document
                      <ArrowRight className="h-5 w-5 ml-3" />
                    </div>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Card className="backdrop-blur-xl bg-red-50/80 dark:bg-red-900/20 border-red-200 dark:border-red-700/50 mb-8 animate-in slide-in-from-top-2">
              <CardContent className="p-6">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
                  <span className="text-red-700 dark:text-red-400 font-medium">Error:</span>
                  <span className="text-red-600 dark:text-red-300 ml-2">{error}</span>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Results Display */}
          {results && (
            <Card className="backdrop-blur-xl bg-green-50/80 dark:bg-green-900/20 border-green-200 dark:border-green-700/50 animate-in slide-in-from-top-2">
              <CardHeader className="pb-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg">
                      <CheckCircle2 className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-2xl text-green-800 dark:text-green-200">
                        Analysis Complete
                      </CardTitle>
                      <CardDescription className="text-green-600 dark:text-green-400">
                        Your document has been successfully analyzed
                      </CardDescription>
                    </div>
                  </div>
                  <Badge className="bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300 hover:bg-green-100 dark:hover:bg-green-800 px-4 py-2">
                    {results.answers.length} answers generated
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {results.questions.map((question, index) => (
                    <div key={index} className="group backdrop-blur-xl bg-white/60 dark:bg-slate-800/60 rounded-2xl p-6 shadow-lg border border-white/20 dark:border-slate-700/50 hover:shadow-xl transition-all duration-300 hover:scale-[1.02]">
                      <div className="mb-6">
                        <div className="flex items-center justify-between mb-4">
                          <Badge variant="outline" className="text-xs font-semibold text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-700 bg-blue-50/50 dark:bg-blue-900/20">
                            Question {index + 1}
                          </Badge>
                          <Button
                            onClick={() => copyToClipboard(results.answers[index], index)}
                            variant="ghost"
                            size="sm"
                            className="opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                          >
                            {copiedIndex === index ? (
                              <CheckCircle2 className="h-4 w-4 text-green-500" />
                            ) : (
                              <Copy className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                        <h3 className="font-semibold text-slate-800 dark:text-white text-lg leading-relaxed">
                          {question}
                        </h3>
                      </div>
                      <Separator className="my-6 bg-slate-200 dark:bg-slate-600" />
                      <div className="prose prose-slate dark:prose-invert max-w-none">
                        <p className="text-slate-700 dark:text-slate-300 leading-relaxed text-base">
                          {results.answers[index]}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 mt-16 backdrop-blur-xl bg-white/50 dark:bg-slate-900/50 border-t border-white/20 dark:border-slate-700/50">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center">
            <p className="text-slate-600 dark:text-slate-400">
              © 2024 HackRx AI. Built with ❤️ for intelligent document analysis.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;