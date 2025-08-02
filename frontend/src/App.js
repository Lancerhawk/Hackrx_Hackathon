import React, { useState } from 'react';
import './App.css';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import { AlertCircle, FileText, Search, Brain, CheckCircle2, Clock } from 'lucide-react';

function App() {
  const [documentUrl, setDocumentUrl] = useState('');
  const [questions, setQuestions] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm border-b text-center justify-center items-center flex">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div className='text-center'>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Semantic Search Platform
              </h1>
              <p className="text-slate-600 mt-1">AI-powered document analysis and question answering</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          

          <Card className="border-none bg-white shadow-xl">
            <CardHeader className="pb-6">
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle className="text-2xl font-bold text-slate-800 mb-2">
                    Document Analysis
                  </CardTitle>
                  <CardDescription className="text-slate-600">
                    Upload a document URL and ask questions to get AI-powered answers
                  </CardDescription>
                </div>
                <Button 
                  onClick={loadDemo} 
                  variant="outline" 
                  className="shrink-0 border-blue-200 text-blue-600 hover:bg-blue-50"
                >
                  Load Demo
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-3">
                    Document URL
                  </label>
                  <Input
                    type="url"
                    value={documentUrl}
                    onChange={(e) => setDocumentUrl(e.target.value)}
                    placeholder="https://example.com/document.pdf"
                    required
                    className="w-full p-4 text-lg border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-3">
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
                    className="w-full p-4 text-lg border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  />
                </div>

                <Button 
                  type="submit" 
                  disabled={loading}
                  className="w-full py-4 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white border-none shadow-lg transition-all duration-200 hover:shadow-xl"
                >
                  {loading ? (
                    <div className="flex items-center justify-center">
                      <Clock className="animate-spin h-5 w-5 mr-3" />
                      Processing Document...
                    </div>
                  ) : (
                    <div className="flex items-center justify-center">
                      <Search className="h-5 w-5 mr-3" />
                      Analyze Document
                    </div>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {error && (
            <Card className="border-red-200 bg-red-50 mt-6">
              <CardContent className="p-6">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
                  <span className="text-red-700 font-medium">Error:</span>
                  <span className="text-red-600 ml-2">{error}</span>
                </div>
              </CardContent>
            </Card>
          )}

          {results && (
            <Card className="border-green-200 bg-green-50 mt-6">
              <CardHeader className="pb-4">
                <div className="flex items-center">
                  <CheckCircle2 className="h-6 w-6 text-green-600 mr-3" />
                  <CardTitle className="text-xl text-green-800">
                    Analysis Complete
                  </CardTitle>
                  <Badge className="ml-auto bg-green-100 text-green-700 hover:bg-green-100">
                    {results.answers.length} answers generated
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {results.questions.map((question, index) => (
                    <div key={index} className="bg-white rounded-lg p-6 shadow-sm border border-slate-200">
                      <div className="mb-4">
                        <Badge variant="outline" className="text-xs font-medium text-blue-600 border-blue-200 mb-3">
                          Question {index + 1}
                        </Badge>
                        <h3 className="font-semibold text-slate-800 text-lg leading-relaxed">
                          {question}
                        </h3>
                      </div>
                      <Separator className="my-4" />
                      <div className="prose prose-slate max-w-none">
                        <p className="text-slate-700 leading-relaxed text-base">
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

      
    </div>
  );
}

export default App;